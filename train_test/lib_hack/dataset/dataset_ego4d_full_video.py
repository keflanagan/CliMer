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


class Ego4dDatasetFullVideo(Dataset):
    def __init__(self, annotations_file, captions_file, same_vid_sampling,
                 metadata, fps, feature_stride, device):
        self.data = pd.read_csv(annotations_file)
        self.full_metadata = pd.read_csv(metadata)
        self.captions_file = captions_file  # path to the lmdb containing the bert features
        self.device = device
        self.same_vid_sampling = same_vid_sampling
        self.fps = fps
        self.feature_stride = feature_stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip1_idx = idx
        narration_clip = self.data['narration'][idx]
        feature_idxs, feature_times, labels, clip_start_stop_time, clip_id, video_id = self.get_feature_idxs(clip1_idx)
        bert_features, num_tokens = self.get_bert_features(clip_id)

        return feature_idxs, feature_times, bert_features, num_tokens, labels, narration_clip, clip_start_stop_time, \
               video_id

    def get_feature_idxs(self, idx):
        """Get the file names of the relevant frames for a clip and add the background frame file names
        to the list
        Returns a list of frame filenames"""
        video_id = self.data['video_id'][idx]
        clip_id = self.data['clip_id'][idx]
        clip_start_frame = self.data['start_frame'][idx]
        clip_stop_frame = self.data['end_frame'][idx]
        clip_start_time = self.data['start_frame'][idx] / self.fps
        clip_stop_time = self.data['end_frame'][idx] / self.fps
        video_durations = self.data['video_duration']
        narration_inds = self.data['narration_ind']

        video_duration = video_durations[idx]
        feature_idx_step = int(1)
        video_end_feature_idx = np.floor(video_duration / (self.feature_stride / self.fps))

        feature_idxs = np.arange(0, video_end_feature_idx, feature_idx_step)
        feature_times = (feature_idxs + 1) * (self.feature_stride / self.fps)

        num_frames = np.floor(video_duration * self.fps)

        feature_locs = np.arange(self.feature_stride - 1, num_frames, self.feature_stride)

        uncombined_labels = []
        for f in feature_locs:
            if f > (clip_stop_frame - 1) or f < (clip_start_frame - 1):
                uncombined_labels.append(0)
            else:
                uncombined_labels.append(1)

        uncombined_labels = torch.tensor(uncombined_labels, dtype=torch.float32)

        clip_start_stop_time = [clip_start_time, clip_stop_time]

        return feature_idxs, feature_times, uncombined_labels, clip_start_stop_time, clip_id, video_id

    def get_bert_features(self, clip_id):
        """Returns the BERT features for each of the captions
        Also returns the number of tokens in the caption in order to ignore padding, cls and sep tokens in the model"""
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
            bert_features = clip["features"].detach()
            num_tokens_clip = clip['num_tokens']
            bert_features = bert_features.squeeze(0).type(torch.float32)

        return bert_features, num_tokens_clip
