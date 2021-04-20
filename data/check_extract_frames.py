import argparse
import glob
import os

parser = argparse.ArgumentParser()
opt = parser.parse_args()
opt.dataset_frames_path = "UCF-101-frames"

video_frame_paths = glob.glob(os.path.join(opt.dataset_frames_path, "*", "*"))

for i, video_frame_path in enumerate(video_frame_paths):
    video_frame_len = len(glob.glob(os.path.join(video_frame_path, "*")))
    if video_frame_len == 0:
        print(i, video_frame_path)

# 49 UCF-101-frames/PlayingGuitar/v_PlayingGuitar_g21_c02
