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
import copy
from utils.utils import common_class, gen_labels, reformat_noun_classes


class EpicDatasetTimestamp(Dataset):
    def __init__(self, annotations_file, metadata, all_data, captions_file, same_vid_sampling,
                 combine, fixed_clip_length, clip_adjacent_timestamps, fps, feature_stride,
                 device):
        self.data = pd.read_csv(annotations_file)
        self.all_data = pd.read_csv(all_data)
        self.full_clip_ids = self.all_data['narration_id']
        self.full_clip_ids_sorted = natsorted(self.full_clip_ids)
        self.metadata = pd.read_csv(metadata)
        self.captions_file = captions_file  # path to the lmdb containing the bert features
        self.device = device
        self.same_vid_sampling = same_vid_sampling
        self.combine = combine
        self.fixed_clip_length = fixed_clip_length
        self.clip_adjacent_timestamps = clip_adjacent_timestamps
        self.fps = fps
        self.feature_stride = feature_stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        clip1_idx, clip2_idx, clip3_idx, clip_bck_idxs, multi_clip_bck_feature_idxs, clip1_feature_idxs, \
            clip2_feature_idxs, video_id, clip1_id, clip2_id, clip3_id, clip_bck_ids, num_bck_id_features, \
            num_features_clip_bck = self.choose_clips(idx)
        narration_clip1 = self.data['narration'][clip1_idx]
        narration_clip2 = self.data['narration'][clip2_idx]
        narration_clip3 = self.data['narration'][clip3_idx]

        combined_video_ids = ['None', 'None']

        if self.combine:
            if self.same_vid_sampling:
                if len(clip_bck_ids) == 0:
                    combined_feature_idxs, clip1_feature_locs = \
                        self.combine_features_no_background(clip1_feature_idxs, clip2_feature_idxs)
                    clip2_feature_locs = []
                else:
                    combined_feature_idxs, clip1_feature_locs, clip2_feature_locs = \
                        self.combine_features_background(clip1_feature_idxs, clip2_feature_idxs,
                                                         multi_clip_bck_feature_idxs)
            else:
                if len(clip_bck_ids) == 0:
                    combined_feature_idxs, combined_video_ids, clip1_feature_locs = \
                        self.combine_features_no_background_diff_videos(clip1_feature_idxs, clip2_feature_idxs,
                                                                        video_ids)
                    clip2_feature_locs = []
                else:
                    combined_feature_idxs, combined_video_ids, clip1_feature_locs, clip2_feature_locs = \
                        self.combine_features_background_diff_videos(clip1_feature_idxs, clip2_feature_idxs,
                                                                     multi_clip_bck_feature_idxs, video_ids,
                                                                     num_features_clip_bck, num_bck_id_features)

            label_caption1, label_caption2, label_caption3, label_no_caption = \
                gen_labels(clip1_feature_locs, clip2_feature_locs)

        else:
            clip1_feature_idxs, video_id = self.get_feature_idxs(clip1_idx)
            feature_idxs, label_caption1, label_caption3, label_no_caption = \
                self.add_background_features_no_combining(clip1_feature_idxs, video_id)
            combined_feature_idxs = feature_idxs
            label_caption2 = torch.tensor([0])
        bert_features, num_tokens = self.get_bert_features(clip1_id, clip2_id, clip3_id)
        caption1, caption2, caption3 = self.get_caption_names(clip1_idx, clip2_idx, clip3_idx)

        combined_feature_idxs = torch.tensor(combined_feature_idxs)
        combined_feature_idxs = combined_feature_idxs.to(torch.long)

        return combined_feature_idxs, bert_features, video_id, combined_video_ids, num_tokens, label_caption1, \
            label_caption2, label_caption3, label_no_caption, (caption1, caption2, caption3), \
            (narration_clip1, narration_clip2, narration_clip3)

    def get_feature_idxs(self, idx):
        """Get the feature idxs for a given clip"""
        video_ids = self.data['video_id']
        clip_ids = self.data['narration_id']
        clip_id = clip_ids[idx]
        video_id = video_ids[idx]
        full_clip_ids = self.full_clip_ids
        full_clip_id_idx = np.where(full_clip_ids == clip_id)[0][0]
        timestamp = self.all_data['narration_timestamp'][full_clip_id_idx]
        full_clip_ids_sorted = self.full_clip_ids_sorted
        unique_video_ids = self.metadata['video_id']
        video_durations = self.metadata['duration']
        video_duration = float(video_durations[np.where(unique_video_ids == video_id)[0][0]])

        # convert time to frame number
        timestamp_split = timestamp.split(':')
        timestamp = float(timestamp_split[0]) * 3600 + float(timestamp_split[1]) * 60 + float(timestamp_split[2])

        # just want to get the indices of the features within the clip

        # account for bad timestamps which are labelled outside the length of the video
        if timestamp > video_duration:
            timestamp = video_duration

        # first need to index based on full metadata list to get ids of surrounding clips
        full_data_clip_index = np.where(np.array(full_clip_ids_sorted) == clip_id)[0][0]

        # need to account for if the clip is the first or last in the entire set
        if full_data_clip_index - 1 < 0:
            full_data_clip_index = len(full_clip_ids_sorted) - 1
        clip_id_before = full_clip_ids_sorted[full_data_clip_index - 1]
        if full_data_clip_index + 1 >= len(full_clip_ids_sorted):
            full_data_clip_index = -1
        clip_id_after = full_clip_ids_sorted[full_data_clip_index + 1]

        # check if the timestamp is not the last in the video
        if clip_id_before.split('_')[:2] == clip_id.split('_')[:2]:
            tb_index = self.all_data.loc[self.all_data['narration_id'] == clip_id_before,
                                              'narration_timestamp'].index.values[0]
            timestamp_before = self.all_data['narration_timestamp'][tb_index]

            timestamp_before_split = timestamp_before.split(':')
            timestamp_before = float(timestamp_before_split[0]) * 3600 + float(timestamp_before_split[1]) * 60 + \
                float(timestamp_before_split[2])
        else:
            timestamp_before = 0

        if clip_id_after.split('_')[:2] == clip_id.split('_')[:2]:
            ta_index = self.all_data.loc[self.all_data['narration_id'] == clip_id_after,
                                              'narration_timestamp'].index.values[0]
            timestamp_after = self.all_data['narration_timestamp'][ta_index]

            timestamp_after_split = timestamp_after.split(':')
            timestamp_after = float(timestamp_after_split[0]) * 3600 + float(timestamp_after_split[1]) * 60 + \
                float(timestamp_after_split[2])
        else:
            timestamp_after = video_duration

        if self.fixed_clip_length == "None" and self.clip_adjacent_timestamps == "None":
            # generate random fraction to determine how much of the video between the previous timestamp and the current
            # timestamp will be included in the segment
            before_frac = random.uniform(0.5, 1)
            artificial_clip_start_time = timestamp - before_frac * (timestamp - timestamp_before)

            # same as above but for the timestamp after
            after_frac = random.uniform(0.5, 1)
            artificial_clip_stop_time = timestamp + after_frac * (timestamp_after - timestamp)

        elif self.clip_adjacent_timestamps == "None":
            half_clip_length = float(self.fixed_clip_length) / 2
            artificial_clip_start_time = np.max((0, timestamp - half_clip_length))
            artificial_clip_stop_time = np.min((timestamp + half_clip_length, video_duration))

        else:
            if self.clip_adjacent_timestamps == "full":
                artificial_clip_start_time = timestamp_before
                artificial_clip_stop_time = timestamp_after
            elif self.clip_adjacent_timestamps == "half":
                artificial_clip_start_time = timestamp - 0.5 * (timestamp - timestamp_before)
                artificial_clip_stop_time = timestamp + 0.5 * (timestamp_after - timestamp)

        # we have a feature every n number of seconds below
        # at feature stride 16 with fps 30
        seconds_per_feature = self.feature_stride / self.fps

        # use this value to determine which features are within the generated clip
        start_feature_value = artificial_clip_start_time / seconds_per_feature
        stop_feature_value = artificial_clip_stop_time / seconds_per_feature

        start_feature_value_round = math.ceil(start_feature_value)
        stop_feature_value_round = math.floor(stop_feature_value)

        # if there is no feature in between the start/stop times then determine what the closest feature is to the
        # segment
        if stop_feature_value_round < start_feature_value_round:
            if (stop_feature_value - stop_feature_value_round) < (start_feature_value_round - start_feature_value):
                start_feature_value_round = stop_feature_value_round
            elif (start_feature_value_round * seconds_per_feature) < video_duration:
                stop_feature_value_round = start_feature_value_round
            else:
                start_feature_value_round = stop_feature_value_round

        start_feature_value_round = np.clip(start_feature_value_round, a_min=1, a_max=None)
        stop_feature_value_round = np.clip(stop_feature_value_round, a_min=1, a_max=None)

        clip_feature_idxs = np.arange(start_feature_value_round - 1, stop_feature_value_round + 1 - 1)

        return clip_feature_idxs, video_id

    def get_first_clip(self, clip1_idx):
        video_ids = self.data['video_id']
        narration_ids = self.data['narration_id']
        verb_class_list = self.data['verb_class']
        noun_classes_list = self.data['all_noun_classes']
        clip1_id = narration_ids[clip1_idx]
        video1_id = video_ids[clip1_idx]

        verb_class_clip1 = verb_class_list[clip1_idx]
        noun_classes_clip1 = noun_classes_list[clip1_idx]
        noun_classes_clip1 = reformat_noun_classes(noun_classes_clip1)

        return video_ids, narration_ids, verb_class_list, noun_classes_list, verb_class_clip1, noun_classes_clip1, \
            clip1_id, video1_id

    def get_second_clip(self, same_video_clips, video_ids, narration_ids, verb_class_list, noun_classes_list,
                        noun_classes_clip1, verb_class_clip1, video1_id, clip1_id):
        count = 0
        while True:  # if the random clip chosen does not match the criteria then reselect
            if self.same_vid_sampling:
                clip2_idx = random.choice(same_video_clips)
            else:
                clip2_idx = int(random.uniform(0, len(video_ids)))
            video2_id = video_ids[clip2_idx]
            clip2_id = narration_ids[clip2_idx]
            verb_class_clip2 = verb_class_list[clip2_idx]
            noun_classes_clip2 = noun_classes_list[clip2_idx]
            noun_classes_clip2 = reformat_noun_classes(noun_classes_clip2)
            # ensure the clips are more than three clips apart and don't have the same verbs or nouns
            # first check if the clips are within the same video

            # restriction changes to just one of the verb/noun being different (so it's not an identical caption)
            if video2_id == video1_id:
                if abs(int(clip2_id.split('_')[-1]) - int(clip1_id.split('_')[-1])) > 3:
                    if not common_class(noun_classes_clip2, noun_classes_clip1) or verb_class_clip2 != \
                            verb_class_clip1:
                        break
                count += 1
                # stop infinite loops - can happen in a video with a really small number of clips
                if count > 20:
                    break

            else:
                if not common_class(noun_classes_clip2, noun_classes_clip1) or verb_class_clip2 != \
                        verb_class_clip1:
                    break
                count += 1
                # stop infinite loops
                if count > 20:
                    break

        return verb_class_clip2, noun_classes_clip2, clip2_idx, clip2_id

    def get_background_clips(self, same_video_clips, video_ids, narration_ids, verb_class_list, noun_classes_list,
                             num_features, noun_classes_clip1, noun_classes_clip2, verb_class_clip1, verb_class_clip2,
                             video1_id, clip1_id, clip2_id):
        clip_bck_idxs = []
        multi_clip_bck_feature_idxs = np.array([])
        clip_bck_ids = []
        # create these lists for comparison with the negative caption
        noun_classes_bck = []
        verb_class_bck = []
        video_bck_ids = []
        num_bck_id_features = []

        while_count = 0
        while num_features < 20:
            while_count += 1
            count = 0
            while True:
                if self.same_vid_sampling:
                    clip_bck_idx = random.choice(same_video_clips)
                else:
                    clip_bck_idx = int(random.uniform(0, len(video_ids)))
                clip_bck_id = narration_ids[clip_bck_idx]
                verb_class_clip_bck = verb_class_list[clip_bck_idx]
                noun_classes_clip_bck = noun_classes_list[clip_bck_idx]
                noun_classes_clip_bck = reformat_noun_classes(noun_classes_clip_bck)

                noun_classes_bck.append(noun_classes_clip_bck)
                verb_class_bck.append(verb_class_clip_bck)

                # ensure that the caption isn't identical (so only one of the conditions needs to hold)
                if (
                        not common_class(noun_classes_clip_bck, noun_classes_clip1) or
                        not common_class(noun_classes_clip_bck, noun_classes_clip2) or
                        verb_class_clip_bck != verb_class_clip1 or verb_class_clip_bck != verb_class_clip2
                ):
                    video_bck_id = video_ids[clip_bck_idx]
                    # if same video check that background is reasonable distance from existing clips
                    if video_bck_id == video1_id:
                        if abs(int(clip_bck_id.split('_')[-1]) - int(clip1_id.split('_')[-1])) > 2 and \
                           abs(int(clip_bck_id.split('_')[-1]) - int(clip2_id.split('_')[-1])) > 2:
                            break
                    break
                count += 1
                # stop infinite loop
                if count > 20:
                    break

            clip_bck_idxs.append(clip_bck_idx)
            clip_bck_ids.append(clip_bck_id)

            # CHECK NUMBER OF FEATURES FROM BACKGROUND CLIP
            clip_bck_feature_idxs, video_bck_id = self.get_feature_idxs(clip_bck_idx)
            video_bck_ids.append(video_bck_id)
            num_bck_id_features.append(len(clip_bck_feature_idxs))

            multi_clip_bck_feature_idxs = np.concatenate((multi_clip_bck_feature_idxs, clip_bck_feature_idxs))

            num_features = num_features + len(clip_bck_feature_idxs)

        return multi_clip_bck_feature_idxs, num_bck_id_features, video_bck_ids, num_features, noun_classes_bck, \
            verb_class_bck, clip_bck_idxs, clip_bck_ids

    @staticmethod
    def get_negative_clip(same_video_clips, narration_ids, verb_class_list, noun_classes_list, noun_classes_clip1,
                          noun_classes_clip2, noun_classes_bck, verb_class_clip1, verb_class_clip2, verb_class_bck):
        count = 0
        while True:  # if the random clip chosen does not match the criteria then reselect
            # clip3_idx = int(random.uniform(0, len(self.clip_ids)))
            clip3_idx = random.choice(same_video_clips)
            clip3_id = narration_ids[clip3_idx]
            verb_class_clip3 = verb_class_list[clip3_idx]
            noun_classes_clip3 = noun_classes_list[clip3_idx]
            noun_classes_clip3 = reformat_noun_classes(noun_classes_clip3)

            # check to make sure it is distinct:
            # check background clips first:
            bck_verb_match = False
            bck_noun_match = False
            if len(verb_class_bck) > 0:
                for j, verb in enumerate(verb_class_bck):
                    if verb_class_clip3 == verb:
                        bck_verb_match = True
                    if common_class(noun_classes_clip3, noun_classes_bck[j]):
                        bck_noun_match = True

            # just a resitriction on how far apart the clips are, not on verbs/nouns, except for ensuring
            # that the caption isn't identical (so only one of the conditions needs to hold)
            if (
                    not common_class(noun_classes_clip3, noun_classes_clip1) or
                    not common_class(noun_classes_clip3, noun_classes_clip2) or
                    verb_class_clip3 != verb_class_clip1 or verb_class_clip3 != verb_class_clip2 or
                    bck_verb_match or bck_noun_match
            ):
                break

            # if it can't find one that satisfies the criteria then just settle
            count += 1
            if count > 20:
                break

        return clip3_idx, clip3_id

    def choose_clips(self, idx):
        """Choose the second clip and random negative caption
        Return indices of each of the clips"""

        ############################
        # GET FIRST CLIP IDX
        clip1_idx = idx

        video_ids, narration_ids, verb_class_list, noun_classes_list, verb_class_clip1, noun_classes_clip1, \
            clip1_id, video1_id = self.get_first_clip(clip1_idx)
        ############################

        # ensure same video_id
        same_video_clips = np.where(video_ids == video1_id)[0]

        #############################
        # GET SECOND CLIP IDX
        verb_class_clip2, noun_classes_clip2, clip2_idx, clip2_id = \
            self.get_second_clip(same_video_clips, video_ids, narration_ids, verb_class_list, noun_classes_list,
                                 noun_classes_clip1, verb_class_clip1, video1_id, clip1_id)
        #############################
        # CHECK NUMBER OF FEATURES FROM CLIP 1 AND CLIP 2
        clip1_feature_idxs, video_id = self.get_feature_idxs(clip1_idx)
        clip2_feature_idxs, video_id = self.get_feature_idxs(clip2_idx)

        num_features_clip1 = len(clip1_feature_idxs)
        num_features_clip2 = len(clip2_feature_idxs)
        num_features = num_features_clip1 + num_features_clip2

        # if you have more than 20 features with 2 clips then sample them to 20 features overall
        if num_features > 20:
            clip1_feature_idxs, clip2_feature_idxs = \
                self.clip_1_and_2_sampling_no_background(clip1_feature_idxs, clip2_feature_idxs)

        # If fewer than 20 features then get features from a background clip
        ################################
        # GET BACKGROUND CLIP IDXS
        multi_clip_bck_feature_idxs, num_bck_id_features, video_bck_ids, num_features, noun_classes_bck, \
            verb_class_bck, clip_bck_idxs, clip_bck_ids = \
            self.get_background_clips(same_video_clips, video_ids, narration_ids, verb_class_list, noun_classes_list,
                                      num_features, noun_classes_clip1, noun_classes_clip2, verb_class_clip1,
                                      verb_class_clip2, video1_id, clip1_id, clip2_id)
        ###########################################

        # only 20 features
        orig_num_bck = len(multi_clip_bck_feature_idxs)
        num_bck_needed = 20 - (num_features_clip1 + num_features_clip2)
        multi_clip_bck_feature_idxs = multi_clip_bck_feature_idxs[:num_bck_needed]
        num_removed = orig_num_bck - num_bck_needed
        num_bck_needed = 20 - (num_features_clip1 + num_features_clip2)
        if num_bck_needed > 0:
            num_features_clip_bck = num_bck_needed
        else:
            num_features_clip_bck = None

        if len(clip_bck_idxs) > 0:
            num_bck_id_features[-1] = num_bck_id_features[-1] - num_removed

        #############################
        # GET NEG CLIP IDXS
        clip3_idx, clip3_id = self.get_negative_clip(same_video_clips, narration_ids, verb_class_list,
                                                     noun_classes_list, noun_classes_clip1, noun_classes_clip2,
                                                     noun_classes_bck, verb_class_clip1, verb_class_clip2,
                                                     verb_class_bck)

        return clip1_idx, clip2_idx, clip3_idx, clip_bck_idxs, multi_clip_bck_feature_idxs, clip1_feature_idxs, \
            clip2_feature_idxs, video_id, clip1_id, clip2_id, clip3_id, clip_bck_ids, \
            num_bck_id_features, num_features_clip_bck

    @staticmethod
    def combine_features_no_background(clip1_sampled_feature_idxs, clip2_sampled_feature_idxs):
        # clip 1 or 2 could be the longer clip so just use clip 1 as the initial one and pad around that with clip 2

        num_features_clip1 = len(clip1_sampled_feature_idxs)
        num_features_clip2 = len(clip2_sampled_feature_idxs)
        clip1_start_idx = np.random.randint(0, num_features_clip2)
        # insert clip 1 at the specified location in clip 2
        combined_feature_idxs = np.insert(clip2_sampled_feature_idxs, clip1_start_idx, clip1_sampled_feature_idxs)

        # return the locations of clip 1 features also for label generation
        clip1_feature_locs = np.arange(clip1_start_idx, clip1_start_idx + num_features_clip1)

        return combined_feature_idxs, clip1_feature_locs

    @staticmethod
    def combine_features_background(clip1_sampled_feature_idxs, clip2_sampled_feature_idxs,
                                    multi_clip_bck_feature_idxs):
        # Randomise the order in which the clips are combined, then add clip 1 at a random location

        num_features_clip1 = len(clip1_sampled_feature_idxs)
        num_features_clip2 = len(clip2_sampled_feature_idxs)
        num_features_bck = len(multi_clip_bck_feature_idxs)

        clip2_start_idx = np.random.randint(0, num_features_bck)

        # insert clip 2 at the specified location in the background clips
        combined_feature_idxs = np.insert(multi_clip_bck_feature_idxs, clip2_start_idx, clip2_sampled_feature_idxs)

        clip2_feature_locs = np.arange(clip2_start_idx, clip2_start_idx + num_features_clip2)

        clip1_start_idx = np.random.randint(0, len(combined_feature_idxs))

        # insert clip 1 at the specified location in the background clips
        combined_feature_idxs = np.insert(combined_feature_idxs, clip1_start_idx, clip1_sampled_feature_idxs)

        # return the locations of clip 1 features also for label generation
        clip1_feature_locs = np.arange(clip1_start_idx, clip1_start_idx + num_features_clip1)

        if np.min(clip1_feature_locs) <= np.max(clip2_feature_locs):
            shift_locs = np.where(clip2_feature_locs >= np.min(clip1_feature_locs))[0]
            clip2_feature_locs[shift_locs] = clip2_feature_locs[shift_locs] + num_features_clip1

        return combined_feature_idxs, clip1_feature_locs, clip2_feature_locs

    @staticmethod
    def combine_features_no_background_diff_videos(clip1_sampled_feature_idxs, clip2_sampled_feature_idxs, video_ids):
        # clip 1 or 2 could be the longer clip so just use clip 1 as the initial one and pad around that with clip 2

        num_features_clip1 = len(clip1_sampled_feature_idxs)
        num_features_clip2 = len(clip2_sampled_feature_idxs)
        clip1_start_idx = np.random.randint(0, num_features_clip2)
        # insert clip 1 at the specified location in clip 2
        combined_feature_idxs = np.insert(clip2_sampled_feature_idxs, clip1_start_idx, clip1_sampled_feature_idxs)
        video1_id = video_ids[0]
        video2_id = video_ids[1]

        video1_ids = np.array([video1_id] * num_features_clip1)
        video2_ids = np.array([video2_id] * num_features_clip2)

        combined_video_ids = np.insert(video2_ids, clip1_start_idx, video1_ids)

        # return the locations of clip 1 features also for label generation
        clip1_feature_locs = np.arange(clip1_start_idx, clip1_start_idx + num_features_clip1)

        return combined_feature_idxs, combined_video_ids, clip1_feature_locs

    @staticmethod
    def combine_features_background_diff_videos(clip1_sampled_feature_idxs, clip2_sampled_feature_idxs,
                                                multi_clip_bck_feature_idxs, video_ids, num_features_clip_bck,
                                                num_bck_id_features):
        # Randomise the order in which the clips are combined, then add clip 1 at a random location
        num_features_clip1 = len(clip1_sampled_feature_idxs)
        num_features_clip2 = len(clip2_sampled_feature_idxs)

        video1_id = video_ids[0]
        video2_id = video_ids[1]
        video_bck_ids = video_ids[2:]
        video_bck_id_list = []

        for i, bck_id in enumerate(video_bck_ids):
            num_features_bck = num_bck_id_features[i]
            video_bck_id_multiple = [bck_id] * num_features_bck
            video_bck_id_list += video_bck_id_multiple

        video_bck_id_list = np.array(video_bck_id_list)

        video1_ids = np.array([video1_id] * num_features_clip1)
        video2_ids = np.array([video2_id] * num_features_clip2)

        clip2_start_idx = np.random.randint(0, num_features_clip_bck)

        # insert clip 2 at the specified location in the background clips
        combined_feature_idxs = np.insert(multi_clip_bck_feature_idxs, clip2_start_idx, clip2_sampled_feature_idxs)

        clip2_feature_locs = np.arange(clip2_start_idx, clip2_start_idx + num_features_clip2)

        clip1_start_idx = np.random.randint(0, len(combined_feature_idxs))

        # insert clip 1 at the specified location in the background clips
        combined_feature_idxs = np.insert(combined_feature_idxs, clip1_start_idx, clip1_sampled_feature_idxs)

        # return the locations of clip 1 features also for label generation
        clip1_feature_locs = np.arange(clip1_start_idx, clip1_start_idx + num_features_clip1)

        # print(video_bck_id_list, clip2_start_idx)
        combined_video_ids = np.insert(video_bck_id_list, clip2_start_idx, video2_ids)
        combined_video_ids = np.insert(combined_video_ids, clip1_start_idx, video1_ids)

        if np.min(clip1_feature_locs) <= np.max(clip2_feature_locs):
            shift_locs = np.where(clip2_feature_locs >= np.min(clip1_feature_locs))[0]
            clip2_feature_locs[shift_locs] = clip2_feature_locs[shift_locs] + num_features_clip1

        return combined_feature_idxs, combined_video_ids, clip1_feature_locs, clip2_feature_locs

    def add_background_features_no_combining(self, clip1_sampled_feature_idxs, video_id):
        """add background to the video clip for training option where video clips are not combined"""

        all_video_ids = np.array(self.metadata['video_id'])
        all_video_num_frames = np.array(self.metadata['num_frames'])
        video_idx = np.where(all_video_ids == video_id)[0]
        num_frames_video = all_video_num_frames[video_idx][0]
        num_features_video = np.ceil((num_frames_video - 16) / 16)

        # clip 1 or 2 could be the longer clip so just use clip 1 as the initial one and pad around that with clip 2

        num_features_clip1 = len(clip1_sampled_feature_idxs)
        if num_features_clip1 >= 5:
            num_sampled_features = np.random.randint(5, 16)
            if num_sampled_features > num_features_clip1:
                num_sampled_features = num_features_clip1

        else:
            num_sampled_features = num_features_clip1

        unrounded_idxs = \
            np.linspace(clip1_sampled_feature_idxs[0], clip1_sampled_feature_idxs[-1], num_sampled_features)
        sampled_positive_feature_idxs = np.round(unrounded_idxs)
        positive_first_index = sampled_positive_feature_idxs[0]
        positive_last_index = sampled_positive_feature_idxs[-1]

        # get the feature rate so that you can match the background feature rate to it
        # make sure that there is more than one idx in the list first - use feature rate = 1 if only 1
        if len(unrounded_idxs) > 1:
            feature_rate = unrounded_idxs[1] - unrounded_idxs[0]
        else:
            feature_rate = 1

        if feature_rate * 20 > num_features_video:
            feature_rate = num_features_video / 20

        num_background = 20 - len(sampled_positive_feature_idxs)

        num_before = np.random.randint(0, num_background + 1)
        num_after = num_background - num_before

        if positive_first_index - num_before * feature_rate < 0:
            num_before = int(np.floor(positive_first_index / feature_rate))
            num_after = num_background - num_before

        if positive_last_index + num_after * feature_rate >= num_features_video:
            num_after = int(np.floor(((num_features_video - 1) - positive_last_index) / feature_rate))
            if num_after < 0:
                num_after = 0
            num_before = num_background - num_after

        segment_start_index = np.floor(positive_first_index - num_before * feature_rate)
        segment_end_index = np.floor(positive_last_index + num_after * feature_rate)

        before_indices = np.round(np.linspace(segment_start_index, positive_first_index, num=num_before+1))[:-1]
        after_indices = np.round(np.linspace(positive_last_index, segment_end_index, num=num_after+1))[1:]
        before_indices = np.clip(before_indices, a_min=0, a_max=None)
        after_indices = np.clip(after_indices, a_min=None, a_max=num_features_video-1)

        final_feature_idxs = np.concatenate((before_indices, sampled_positive_feature_idxs, after_indices))

        # generate labels
        positive_clip_start_index = len(before_indices)
        positive_clip_end_index = positive_clip_start_index + len(sampled_positive_feature_idxs) - 1
        clip_feature_locs = np.arange(positive_clip_start_index, positive_clip_end_index + 1)
        clip_feature_locs = torch.from_numpy(clip_feature_locs)

        label_caption1 = np.zeros(20)
        assert clip_feature_locs[-1] <= 19
        label_caption1[clip_feature_locs] = 1
        label_caption3 = np.zeros(20)
        label_no_caption = np.ones(20) * 0.5

        label_caption1 = torch.tensor(label_caption1, dtype=torch.float32)
        label_caption3 = torch.tensor(label_caption3, dtype=torch.float32)
        label_no_caption = torch.tensor(label_no_caption, dtype=torch.float32)

        return final_feature_idxs, label_caption1, label_caption3, label_no_caption

    @staticmethod
    def clip_1_and_2_sampling_no_background(clip1_feature_idxs, clip2_feature_idxs):
        clip1_length = len(clip1_feature_idxs)
        clip2_length = len(clip2_feature_idxs)

        longest = np.argmax((clip1_length, clip2_length))
        if longest == 0:
            longer_clip_idxs = clip1_feature_idxs
            shorter_clip_idxs = clip2_feature_idxs
        else:
            longer_clip_idxs = clip2_feature_idxs
            shorter_clip_idxs = clip1_feature_idxs

        # if shorter clip is longer than 5 features then sample randomly, otherwise just sample 20 - number of features
        # in the shorter clip
        if len(shorter_clip_idxs) > 5:
            # account for id there are more than 20 features in the shorter clip such that you have at least 5 features
            # from the longer clip
            min_longer = np.max([5, 20 - len(shorter_clip_idxs)])
            num_longer_clip_features = np.random.randint(min_longer, 15)
            longer_clip_sample_idxs = \
                np.round(np.linspace(longer_clip_idxs[0], longer_clip_idxs[-1], num_longer_clip_features))

            num_shorter_clip_features = 20 - num_longer_clip_features
            shorter_clip_sample_idxs = \
                np.round(np.linspace(shorter_clip_idxs[0], shorter_clip_idxs[-1], num_shorter_clip_features))

        else:
            num_longer_clip_features = 20 - len(shorter_clip_idxs)
            longer_clip_sample_idxs = \
                np.round(np.linspace(longer_clip_idxs[0], longer_clip_idxs[-1], num_longer_clip_features))

            shorter_clip_sample_idxs = shorter_clip_idxs

        if longest == 0:
            clip1_sampled_feature_idxs = longer_clip_sample_idxs
            clip2_sampled_feature_idxs = shorter_clip_sample_idxs
        else:
            clip1_sampled_feature_idxs = shorter_clip_sample_idxs
            clip2_sampled_feature_idxs = longer_clip_sample_idxs

        return clip1_sampled_feature_idxs, clip2_sampled_feature_idxs

    def get_bert_features(self, clip1_id, clip2_id, clip3_id):
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
        clip1_id = str(clip1_id).encode("ascii")
        clip2_id = str(clip2_id).encode("ascii")
        clip3_id = str(clip3_id).encode("ascii")
        with caption_data.begin(write=False) as txn:
            clip1 = pickle.loads(txn.get(clip1_id))
            bert_features1 = clip1["features"].detach().to('cpu')
            num_tokens_clip1 = clip1['num_tokens']
            bert_features1 = bert_features1.squeeze(0).type(torch.float32)

            clip2 = pickle.loads(txn.get(clip2_id))
            bert_features2 = clip2["features"].detach().to('cpu')
            num_tokens_clip2 = clip2['num_tokens']
            bert_features2 = bert_features2.squeeze(0).type(torch.float32)

            clip3 = pickle.loads(txn.get(clip3_id))
            bert_features3 = clip3["features"].detach().to('cpu')
            num_tokens_clip3 = clip3['num_tokens']
            bert_features3 = bert_features3.squeeze(0).type(torch.float32)

        return (bert_features1, bert_features2, bert_features3), (num_tokens_clip1, num_tokens_clip2, num_tokens_clip3)

    def get_caption_names(self, clip1_idx, clip2_idx, clip3_idx):
        caption1 = self.data['narration'][clip1_idx]
        caption2 = self.data['narration'][clip2_idx]
        caption3 = self.data['narration'][clip3_idx]

        return caption1, caption2, caption3
