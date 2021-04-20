import argparse
import io
import logging
import os
from pathlib import Path

import hydra
import skvideo.io
import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from data.extract_frames import extract_frames
from dataset import *
from models import *


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


@hydra.main(config_path="./configs", config_name="default.yaml")
def main(cfg: DictConfig) -> None:

    if "experiments" in cfg.keys():
        cfg = OmegaConf.merge(cfg, cfg.experiments)

    # A logger for this file
    logger = logging.getLogger(__name__)

    # NOTE: hydra causes the python file to run in hydra.run.dir by default
    logger.info(f"Run script in {HydraConfig.get().run.dir}")

    dataset_path = (Path(cfg.dataset.root) / cfg.dataset.frames).as_posix()
    video_path = (
        Path(cfg.dataset.root) / cfg.dataset.name / cfg.test.video_name
    ).as_posix()
    save_gif_name = Path(cfg.test.video_name).stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (cfg.test.channels, cfg.test.image_height, cfg.test.image_width)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    labels = sorted(list(set(os.listdir(dataset_path))))

    # Define model and load model checkpoint
    model = CNNLSTM(
        num_classes=cfg.test.num_classes,
        latent_dim=cfg.test.latent_dim,
        lstm_layers=cfg.test.lstm_layers,
        hidden_dim=cfg.test.hidden_dim,
        bidirectional=cfg.test.bidirectional,
        attention=cfg.test.attention,
    )
    ckpt = torch.load(cfg.test.checkpoint_model, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Extract predictions
    output_frames = []
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for frame in tqdm.tqdm(
        extract_frames(video_path), file=tqdm_out, desc="Processing frames"
    ):
        image_tensor = Variable(transform(frame)).to(device)
        image_tensor = image_tensor.view(1, 1, *image_tensor.shape)

        # Get label prediction for frame
        with torch.no_grad():
            prediction = model(image_tensor)
            predicted_label = labels[prediction.argmax(1).item()]

        # Draw label on frame
        d = ImageDraw.Draw(frame)
        d.text(xy=(10, 10), text=predicted_label, fill=(255, 255, 255))

        output_frames += [frame]

    # Create video from frames
    writer = skvideo.io.FFmpegWriter(f"{save_gif_name}.gif")
    for frame in tqdm.tqdm(output_frames, file=tqdm_out, desc="Writing to video"):
        writer.writeFrame(np.array(frame))
    writer.close()


if __name__ == "__main__":
    main()
