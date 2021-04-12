import argparse
import glob
import logging
import os
from pathlib import Path

import hydra
import skvideo.io
import tqdm
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from data.extract_frames import extract_frames
from dataset import *
from models import *


@hydra.main(config_path="./configs", config_name="default.yaml")
def main(cfg):

    if "experiments" in cfg.keys():
        cfg = OmegaConf.merge(cfg, cfg.experiments)

    dataset_path = (Path(cfg.dataset.root) / cfg.dataset.frames).as_posix()
    video_path = (
        Path(cfg.dataset.root) / cfg.dataset.name / cfg.test.video_name
    ).as_posix()
    save_gif_name = Path(cfg.test.video_name).stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (cfg.test.channels, cfg.test.image_dim, cfg.test.image_dim)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    labels = sorted(list(set(os.listdir(dataset_path))))

    # Define model and load model checkpoint
    model = CNNLSTM(num_classes=len(labels), latent_dim=cfg.test.latent_dim)
    model.to(device)
    model.load_state_dict(torch.load(cfg.test.checkpoint_model))
    model.eval()

    # Extract predictions
    output_frames = []
    for frame in tqdm.tqdm(extract_frames(video_path), desc="Processing frames"):
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
    for frame in tqdm.tqdm(output_frames, desc="Writing to video"):
        writer.writeFrame(np.array(frame))
    writer.close()


if __name__ == "__main__":
    main()
