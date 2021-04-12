#!/bin/bash

# Downloads the UCF-101 dataset
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
# Unzip the UCF-101 dataset
unrar x UCF101.rar
unzip UCF101TrainTestSplits-RecognitionTask.zip
