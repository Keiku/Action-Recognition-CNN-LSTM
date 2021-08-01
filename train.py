import argparse
import datetime
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.timer import Timer
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from models import *


@hydra.main(config_path="./configs", config_name="default.yaml")
def main(cfg: DictConfig) -> None:

    if "experiments" in cfg.keys():
        cfg = OmegaConf.merge(cfg, cfg.experiments)

    if "debug" in cfg.keys():
        logger.info(f"Run script in debug")
        cfg = OmegaConf.merge(cfg, cfg.debug)

    # A logger for this file
    logger = logging.getLogger(__name__)

    # NOTE: hydra causes the python file to run in hydra.run.dir by default
    logger.info(f"Run script in {HydraConfig.get().run.dir}")

    writer = SummaryWriter(log_dir=cfg.train.tensorboard_dir)

    checkpoints_dir = Path(cfg.train.checkpoints_dir)
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_shape = (cfg.train.channels, cfg.train.image_height, cfg.train.image_width)

    # NOTE: With hydra, the python file runs in hydra.run.dir by default, so set the dataset path to a full path or an appropriate relative path
    dataset_path = Path(cfg.dataset.root) / cfg.dataset.frames
    split_path = Path(cfg.dataset.root) / cfg.dataset.split_file
    assert dataset_path.exists(), "Video image folder not found"
    assert (
        split_path.exists()
    ), "The file that describes the split of train/test not found."

    # Define training set
    train_dataset = Dataset(
        dataset_path=dataset_path,
        split_path=split_path,
        split_number=cfg.dataset.split_number,
        input_shape=image_shape,
        sequence_length=cfg.train.sequence_length,
        training=True,
    )

    # Define train dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )

    # Define test set
    test_dataset = Dataset(
        dataset_path=dataset_path,
        split_path=split_path,
        split_number=cfg.dataset.split_number,
        input_shape=image_shape,
        sequence_length=cfg.train.sequence_length,
        training=False,
    )

    # Define test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # Classification criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Define network
    model = CNNLSTM(
        num_classes=train_dataset.num_classes,
        latent_dim=cfg.train.latent_dim,
        lstm_layers=cfg.train.lstm_layers,
        hidden_dim=cfg.train.hidden_dim,
        bidirectional=cfg.train.bidirectional,
        attention=cfg.train.attention,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    checkpointer = Checkpointer(
        model,
        optimizer=optimizer,
        # scheduler=scheduler,
        save_dir=cfg.train.checkpoints_dir,
        save_to_disk=True,
    )

    if cfg.train.resume:
        if not checkpointer.has_checkpoint():
            start_epoch = 0
        else:
            ckpt = checkpointer.resume_or_load("", resume=True)
            start_epoch = ckpt["epoch"]
            model.to(device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    elif cfg.train.checkpoint_model != "":
        ckpt = torch.load(cfg.train.checkpoint_model, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, cfg.train.num_epochs):
        epoch += 1
        epoch_metrics = {"loss": [], "acc": []}
        timer = Timer()
        for batch_i, (X, y) in enumerate(train_dataloader):
            batch_i += 1
            if X.size(0) == 1:
                continue

            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()

            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            # Get sequence predictions
            predictions = model(image_sequences)

            # Compute metrics
            loss = criterion(predictions, labels)
            acc = (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward()
            optimizer.step()

            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = (epoch - 1) * len(train_dataloader) + (batch_i - 1)
            batches_left = cfg.train.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * timer.seconds())
            time_iter = round(timer.seconds(), 3)
            timer.reset()

            logger.info(
                f'Training - [Epoch: {epoch}/{cfg.train.num_epochs}] [Batch: {batch_i}/{len(train_dataloader)}] [Loss: {np.mean(epoch_metrics["loss"]):.3f}] [Acc: {np.mean(epoch_metrics["acc"]):.3f}] [ETA: {time_left}] [Iter time: {time_iter}s/it]'
            )

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        writer.add_scalar("train/loss", np.mean(epoch_metrics["loss"]), epoch)
        writer.add_scalar("train/acc", np.mean(epoch_metrics["acc"]), epoch)

        def test_model(epoch):
            """ Evaluate the model on the test set """
            model.eval()
            test_metrics = {"loss": [], "acc": []}
            timer = Timer()
            for batch_i, (X, y) in enumerate(test_dataloader):
                batch_i += 1
                image_sequences = Variable(X.to(device), requires_grad=False)
                labels = Variable(y, requires_grad=False).to(device)

                with torch.no_grad():
                    # Reset LSTM hidden state
                    model.lstm.reset_hidden_state()
                    # Get sequence predictions
                    predictions = model(image_sequences)

                # Compute metrics
                loss = criterion(predictions, labels)
                acc = (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

                # Keep track of loss and accuracy
                test_metrics["loss"].append(loss.item())
                test_metrics["acc"].append(acc)

                # Determine approximate time left
                batches_done = batch_i - 1
                batches_left = len(test_dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * timer.seconds())
                time_iter = round(timer.seconds(), 3)
                timer.reset()

                # Log test performance
                logger.info(
                    f'Testing - [Epoch: {epoch}/{cfg.train.num_epochs}] [Batch: {batch_i}/{len(test_dataloader)}] [Loss: {np.mean(test_metrics["loss"]):.3f}] [Acc: {np.mean(test_metrics["acc"]):.3f}] [ETA: {time_left}] [Iter time: {time_iter}s/it]'
                )

            writer.add_scalar("test/loss", np.mean(test_metrics["loss"]), epoch)
            writer.add_scalar("test/acc", np.mean(test_metrics["acc"]), epoch)

            model.train()

        # Evaluate the model on the test set
        test_model(epoch)

        # Save model checkpoint
        if epoch % cfg.train.checkpoint_interval == 0:
            checkpointer.save(f"checkpoint_{epoch:04}", epoch=epoch)

    writer.close()


if __name__ == "__main__":
    main()
