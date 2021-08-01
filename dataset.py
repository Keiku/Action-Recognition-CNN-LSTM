import time
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split_path,
        split_number,
        input_shape,
        sequence_length,
        training,
    ):
        self.label_mapping = self._extract_label_mapping(split_path)
        self.sequence_paths = self._extract_sequence_paths(
            dataset_path, split_path, split_number, training
        )
        self.label_names = sorted(
            list(set([self._activity_from_path(seq_path) \
                      for seq_path in self.sequence_paths]))
        )
        self.num_classes = len(self.label_names)
        self.sequence_length = sequence_length
        self.training = training
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _extract_label_mapping(self, split_path):
        """ Extracts a mapping between activity name and softmax index """

        # read classInd.txt
        split_path = Path(split_path)
        label_path = split_path / "classInd.txt"
        with open(label_path.as_posix()) as f:
            lines = f.read().splitlines()

        # create label mappings
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1

        return label_mapping

    def _extract_sequence_paths(
        self,
        dataset_path,
        split_path,
        split_number=1,
        training=True,
    ):
        """ Extracts paths to sequences given the specified train / test split """
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        split_file = (
            f"trainlist0{split_number}.txt"
            if training
            else f"testlist0{split_number}.txt"
        )
        split_path = Path(split_path)
        split_file_path = split_path / split_file

        # read split file
        with open(split_file_path.as_posix()) as f:
            lines = f.read().splitlines()

        # delete no frames
        if "PlayingGuitar/v_PlayingGuitar_g21_c02.avi 63" in lines:
            lines.remove("PlayingGuitar/v_PlayingGuitar_g21_c02.avi 63")

        # create sequence paths
        sequence_paths = []
        for line in lines:
            action_video_name = Path(line.split(" ")[0])
            seq_name = action_video_name.with_suffix("")
            sequence_paths += [(Path(dataset_path) / seq_name).as_posix()]

        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        activity = path.split("/")[-2]
        return activity

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        frame_number = int(Path(image_path).stem)
        return frame_number

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        # Set index in mini-batch
        sequence_path = self.sequence_paths[index]

        # Sort frame sequence based on frame number
        image_paths = sorted(
            glob.glob(f"{sequence_path}/*.jpg"),
            key=lambda path: self._frame_number(path),
        )

        # Pad frames sequences shorter than `self.sequence_length` to length
        image_paths = self._pad_to_length(image_paths)

        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(
                1, len(image_paths) // self.sequence_length + 1
            )
            start_i = np.random.randint(
                0, len(image_paths) - sample_interval * self.sequence_length + 1
            )
            flip = np.random.random() < 0.5
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = (
                1
                if self.sequence_length is None
                else len(image_paths) // self.sequence_length
            )
            flip = False

        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            # Append up to sequence_length
            if (
                self.sequence_length is None
                or len(image_sequence) < self.sequence_length
            ):
                image = Image.open(image_paths[i])
                image_tensor = self.transform(image)
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        image_sequence = torch.stack(image_sequence)
        target = self.label_mapping[self._activity_from_path(sequence_path)]

        return image_sequence, target

    def __len__(self):
        return len(self.sequence_paths)
