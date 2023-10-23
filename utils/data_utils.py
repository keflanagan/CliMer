import pandas as pd
import numpy as np
import torch

def load_omnivore_clip_features(omnivore_dict, combined_feature_idxs, video_id):
    batch_features = torch.empty(0)
    combined_feature_idxs = combined_feature_idxs.to(torch.long)
    for i, batch_video_id in enumerate(video_id):
        file = omnivore_dict[batch_video_id]

        try:
            features = file[combined_feature_idxs[i] - 1, :]
        except:
            if torch.max(combined_feature_idxs[i] - 1) >= file.size()[0]:
                combined_feature_idxs[i] = torch.clamp(combined_feature_idxs[i], max=file.size()[0]-1)
                features = file[combined_feature_idxs[i], :]

        features = features.unsqueeze(0)
        batch_features = torch.cat((batch_features, features))

    batch_features = batch_features.to('cpu')

    return batch_features

def load_omnivore_clip_features_diff_videos(omnivore_dict, combined_feature_idxs, combined_video_ids):
    batch_features = torch.empty(0)
    combined_feature_idxs = combined_feature_idxs.to(torch.long)
    # loads all elements in the batch so need to go through the batch one by one
    for i, batch_video_id in enumerate(combined_video_ids):
        feature_idxs = combined_feature_idxs[i] - 1
        features = torch.tensor([])
        for j, vid_id in enumerate(batch_video_id):

            feature_idx = feature_idxs[j]
            file = omnivore_dict[vid_id]

            try:
                feature = file[feature_idx, :].unsqueeze(0)
                features = torch.cat((features, feature))
            except:
                if feature_idx >= file.size()[0]:
                    feature_idx = file.size()[0] - 1
                    feature = file[feature_idx, :].unsqueeze(0)
                    features = torch.cat((features, feature))
                else:
                    print('combined_feature_idxs', combined_feature_idxs)
                    print('DIFFERENT PROBLEM')

        features = features.unsqueeze(0)
        batch_features = torch.cat((batch_features, features))

    batch_features = batch_features.to('cpu')

    return batch_features

def load_all_omnivore_features(caption_data_train, caption_data_val, caption_data_test, omnivore_features_path):
    train_data = pd.read_csv(caption_data_train)
    val_data = pd.read_csv(caption_data_val)
    test_data = pd.read_csv(caption_data_test)
    
    train_ids = list(set(np.array(train_data['video_id'])))
    val_ids = list(set(np.array(val_data['video_id'])))
    test_ids = list(set(np.array(test_data['video_id'])))
    all_ids = train_ids + val_ids + test_ids

    omnivore_dict = {}

    for i, vid_id in enumerate(all_ids):
        if i % 100 == 0:
            print(i)
        file = torch.load(
            f'{omnivore_features_path}{vid_id}.pt', map_location='cpu')
        file.to('cpu')
        omnivore_dict[vid_id] = file

    return omnivore_dict
