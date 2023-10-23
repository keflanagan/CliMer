import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import pandas as pd
import os
from gulpio2 import GulpDirectory
import random
import numpy as np
import lmdb
import pickle
import time
from natsort import natsorted
import math


class EpicDatasetFullVideo(Dataset):
    def __init__(self, annotations_file, captions_file, same_vid_sampling,
                 metadata, fps, feature_stride, device):
        self.data = pd.read_csv(annotations_file)
        self.metadata = pd.read_csv(metadata)
        self.captions_file = captions_file  # path to the lmdb containing the bert features
        self.device = device
        self.same_vid_sampling = same_vid_sampling
        self.fps = fps
        self.feature_stride = feature_stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get clip ids, generate the frames for each video, combine, and then generate the features and labels
        clip1_idx = idx
        narration_clip = self.data['narration'][idx]
        feature_idxs, feature_times, labels, clip_start_stop_time, clip_id, video_id = self.get_frame_names(clip1_idx)
        bert_features, num_tokens = self.get_bert_features(clip_id)

        return feature_idxs, feature_times, bert_features, num_tokens, labels, narration_clip, clip_start_stop_time, \
            video_id

    def get_frame_names(self, idx):
        """Get the file names of the relevant frames for a clip and add the background frame file names
        to the list
        Returns a list of frame filenames"""
        video_id = self.data['video_id'][idx]
        clip_start_frame = self.data['start_frame'][idx]
        clip_stop_frame = self.data['stop_frame'][idx]
        clip_start_time = self.data['start_timestamp'][idx]
        clip_stop_time = self.data['stop_timestamp'][idx]
        unique_vid_ids = np.array(self.metadata['video_id'])
        actual_fps_values = self.metadata['fps']
        actual_fps = actual_fps_values[np.where(unique_vid_ids == video_id)[0][0]]
        video_durations = self.metadata['duration']

        clip_ids = self.data['narration_id']
        clip_id = clip_ids[idx]

        video_duration = video_durations[np.where(unique_vid_ids == video_id)[0][0]]

        feature_idx_step = int(1)

        video_end_feature_idx = np.floor(video_duration / (self.feature_stride / self.fps))

        feature_idxs = np.arange(0, video_end_feature_idx, feature_idx_step)
        feature_times = (feature_idxs + 1) * (self.feature_stride / self.fps)

        num_frames = np.floor(video_duration * actual_fps)
        actual_feature_stride = (self.feature_stride / self.fps) * actual_fps

        feature_locs = np.arange(actual_feature_stride - 1, num_frames, actual_feature_stride)

        #########################
        # create a list denoting whether each frame is background or not
        uncombined_labels = []
        for f in feature_locs:
            if f > (clip_stop_frame - 1) or f < (clip_start_frame - 1):
                uncombined_labels.append(0)
            else:
                uncombined_labels.append(1)

        uncombined_labels = torch.tensor(uncombined_labels, dtype=torch.float32)

        clip_start_time_split = clip_start_time.split(':')
        clip_start_time = float(clip_start_time_split[0]) * 3600 + float(clip_start_time_split[1]) * 60 + \
            float(clip_start_time_split[2])
        clip_stop_time_split = clip_stop_time.split(':')
        clip_stop_time = float(clip_stop_time_split[0]) * 3600 + float(clip_stop_time_split[1]) * 60 + \
            float(clip_stop_time_split[2])

        clip_start_stop_time = [clip_start_time, clip_stop_time]

        return feature_idxs, feature_times, uncombined_labels, clip_start_stop_time, clip_id, video_id

    def get_bert_features(self, clip_id):
        """Returns the BERT features for each of the captions
        Also returns the number of tokens in the caption in order to ignore padding, cls and sep tokens in the model"""
        # pseudo code at the moment as need to generate the BERT features and store them somehow
        # update this when the bert feature file is created
        caption_data = lmdb.open(
            self.captions_file,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        clip_id = str(clip_id).encode("ascii")
        with caption_data.begin(write=False) as txn:
            clip = pickle.loads(txn.get(clip_id))
            bert_features = clip["features"].detach().to('cpu')
            num_tokens_clip = clip['num_tokens']
            bert_features = bert_features.squeeze(0).type(torch.float32)

        return bert_features, num_tokens_clip
