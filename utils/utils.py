import numpy as np
import torch

def iou_stats(iou_time_list):
    iou_time_above_1 = torch.where(iou_time_list > 0.1)
    frac_time_above_1 = iou_time_above_1[0].size()[0] / iou_time_list.size()[0]

    iou_time_above_3 = torch.where(iou_time_list > 0.3)
    frac_time_above_3 = iou_time_above_3[0].size()[0] / iou_time_list.size()[0]

    iou_time_above_5 = torch.where(iou_time_list > 0.5)
    frac_time_above_5 = iou_time_above_5[0].size()[0] / iou_time_list.size()[0]

    iou_time_above_7 = torch.where(iou_time_list > 0.7)
    frac_time_above_7 = iou_time_above_7[0].size()[0] / iou_time_list.size()[0]

    print('IoU 0.1: ', frac_time_above_1)
    print('IoU 0.3: ', frac_time_above_3)
    print('IoU 0.5: ', frac_time_above_5)
    print('IoU 0.7: ', frac_time_above_7)

    return iou_time_above_1, iou_time_above_3, iou_time_above_5, iou_time_above_7, frac_time_above_1, \
        frac_time_above_3, frac_time_above_5, frac_time_above_7

def iou_time(pred_start_time, pred_end_time, start_time, end_time):
    union = torch.max(pred_end_time, end_time) - torch.min(pred_start_time, start_time)
    intersection = torch.min(pred_end_time, end_time) - torch.max(pred_start_time, start_time)
    intersection = torch.max(torch.tensor(0), intersection)
    iou = torch.div(intersection, union)
    iou = torch.unsqueeze(iou, dim=0)

    return iou


def iou_time_based(final_preds, feature_times, clip_start_stop_time):
    # time based iou calculation
    start_time = clip_start_stop_time[0]
    stop_time = clip_start_stop_time[1]
    feature_times = feature_times.squeeze(0)
    # get preds converted into start and end times
    pos_pred_locs = torch.where(final_preds == 1)[0]
    pred_start_feature_idx = torch.min(pos_pred_locs)
    pred_stop_feature_idx = torch.max(pos_pred_locs)
    pred_start_feature_time = feature_times[pred_start_feature_idx]
    pred_stop_feature_time = feature_times[pred_stop_feature_idx]
    try:
        pred_start_time = (pred_start_feature_time + feature_times[pred_start_feature_idx - 1]) / 2
    except:
        pred_start_time = (pred_start_feature_time + 0) / 2
    try:
        pred_stop_time = (pred_stop_feature_time + feature_times[pred_stop_feature_idx + 1]) / 2
    except:
        pred_stop_time = (pred_stop_feature_time + pred_stop_feature_time) / 2

    iou_time_value = iou_time(pred_start_time, pred_stop_time, start_time, stop_time)

    return iou_time_value


def get_hard_preds(preds, threshold):
    preds = preds.squeeze(0)

    # MIN MAX NORMALISATION
    pred_list = (preds - torch.min(preds)) / (torch.max(preds) - torch.min(preds))
    max_pred = torch.max(pred_list)
    pred_list_hard = torch.zeros(pred_list.size(0))
    for j, pred in enumerate(pred_list):
        if pred < max_pred * threshold:
            pred_list_hard[j] = 0
        else:
            pred_list_hard[j] = 1

    segments = [[]]
    pos_features = np.where(pred_list_hard == 1)[0]
    count = 0
    for j, frame in enumerate(pos_features):
        if j == 0 or frame - pos_features[j - 1] == 1:
            segments[count].append(frame)
        else:
            count += 1
            segments.append([])
            segments[count].append(frame)

    max_value = 0
    for j, segment in enumerate(segments):
        seg_max_value = torch.max(pred_list[segment])
        if j == 0 or seg_max_value > max_value:
            max_segment_idx = j
            max_value = seg_max_value

    final_preds = np.zeros(pred_list.size(0))
    final_preds[segments[max_segment_idx]] = 1
    final_preds = torch.tensor(final_preds)

    return final_preds


def split_features(features, seg_size, overlap):
    num_tokens = features.size()[1]

    segment_list = []
    feature_idx_list = []

    num_segments = int(np.ceil((num_tokens - seg_size) / (seg_size - overlap) + 1))

    for i in range(0, num_segments):
        idx_start = (seg_size - overlap) * i
        idx_end = idx_start + seg_size - 1

        # if the last window, then take the last <segment_size> features
        if idx_end > num_tokens:
            idx_end = num_tokens - 1
            idx_start = idx_end - seg_size + 1

        segment = features[:, int(idx_start):int(idx_end + 1)]
        feature_idxs = np.arange(idx_start, idx_end + 1)
        segment_list.append(segment)
        feature_idx_list.append(feature_idxs)

    return segment_list, feature_idx_list


def recombine_preds(preds_list, feature_idx_list, num_features):
    full_preds_list = []
    full_feature_idx_list = np.array([])
    for pred_list in preds_list:
        full_preds_list += pred_list[0]

    for feature_idxs in feature_idx_list:
        full_feature_idx_list = np.concatenate((full_feature_idx_list, feature_idxs))

    full_preds_list = np.array(full_preds_list)
    final_preds_list = np.zeros(num_features)

    for feature_idx in range(0, num_features):
        idxs = np.where(full_feature_idx_list == feature_idx)
        actual_value = np.max(full_preds_list[idxs])
        final_preds_list[feature_idx] = actual_value

    return final_preds_list


def get_weights_single_caption(preds_cap_pos_loc, preds_cap_neg_loc, weights_cap, label_caption, preds_cap, i):
    cap_pos_indices = preds_cap_pos_loc[:, 1][torch.where(preds_cap_pos_loc[:, 0] == i)]
    cap_neg_indices = preds_cap_neg_loc[:, 1][torch.where(preds_cap_neg_loc[:, 0] == i)]

    weights_cap[i, cap_pos_indices] = cap_neg_indices.size()[0] / \
        torch.tensor((cap_neg_indices.size()[0] + cap_pos_indices.size()[0]))

    weights_cap[i, cap_neg_indices] = cap_pos_indices.size()[0] / \
        torch.tensor((cap_neg_indices.size()[0] + cap_pos_indices.size()[0]))

    loss_size_retention_factor_cap = label_caption.size()[1] / \
        (((cap_pos_indices.size()[0] / label_caption.size()[1]) *
         cap_neg_indices.size()[0]) * 2)

    return weights_cap, loss_size_retention_factor_cap


def get_balanced_loss_single_caption(label_caption1, preds_cap1, criterion_no_reduction, device):
    weights_cap1 = torch.zeros(label_caption1.size()).to(device)
    preds_cap1_pos_loc = torch.nonzero(label_caption1)
    preds_cap1_neg_loc = (label_caption1 == 0).nonzero().squeeze(1)

    loss_size_retention_factor_cap1 = []

    for i in range(label_caption1.size()[0]):
        weights_cap1, loss_size_retention_factor_cap1_part = \
            get_weights_single_caption(preds_cap1_pos_loc, preds_cap1_neg_loc, weights_cap1, label_caption1,
                                       preds_cap1, i)

        loss_size_retention_factor_cap1.append(loss_size_retention_factor_cap1_part)

    loss_size_retention_factor_cap1 = torch.tensor(loss_size_retention_factor_cap1).unsqueeze(1).to(device)
    individual_loss_cap1 = criterion_no_reduction(preds_cap1, label_caption1)
    loss_cap1 = torch.mean(
        torch.mul(torch.mul(weights_cap1, individual_loss_cap1), loss_size_retention_factor_cap1))

    return loss_cap1


def get_balanced_loss_batch(label_caption1, label_caption2, preds_cap1, preds_cap2, criterion_no_reduction, device):
    # positive
    weights_cap1 = torch.zeros(label_caption1.size()).to(device)
    preds_cap1_pos_loc = torch.nonzero(label_caption1)
    preds_cap1_neg_loc = (label_caption1 == 0).nonzero().squeeze(1)

    weights_cap2 = torch.zeros(label_caption2.size()).to(device)
    preds_cap2_pos_loc = torch.nonzero(label_caption2)
    preds_cap2_neg_loc = (label_caption2 == 0).nonzero().squeeze(1)

    loss_size_retention_factor_cap1 = []
    loss_size_retention_factor_cap2 = []

    for i in range(label_caption1.size()[0]):
        weights_cap1, loss_size_retention_factor_cap1_part = \
            get_weights_single_caption(preds_cap1_pos_loc, preds_cap1_neg_loc, weights_cap1, label_caption1,
                                            preds_cap1, i)

        weights_cap2, loss_size_retention_factor_cap2_part = \
            get_weights_single_caption(preds_cap2_pos_loc, preds_cap2_neg_loc, weights_cap2, label_caption2,
                                            preds_cap2, i)

        loss_size_retention_factor_cap1.append(loss_size_retention_factor_cap1_part)
        loss_size_retention_factor_cap2.append(loss_size_retention_factor_cap2_part)

    loss_size_retention_factor_cap1 = torch.tensor(loss_size_retention_factor_cap1).unsqueeze(1).to(device)
    loss_size_retention_factor_cap2 = torch.tensor(loss_size_retention_factor_cap2).unsqueeze(1).to(device)
    individual_loss_cap1 = criterion_no_reduction(preds_cap1, label_caption1)

    loss_cap1 = torch.mean(
        torch.mul(torch.mul(weights_cap1, individual_loss_cap1), loss_size_retention_factor_cap1))
    individual_loss_cap2 = criterion_no_reduction(preds_cap2, label_caption2)
    loss_cap2 = torch.mean(
        torch.mul(torch.mul(weights_cap2, individual_loss_cap2), loss_size_retention_factor_cap2))

    return loss_cap1, loss_cap2


def common_class(a, b):
    """For comparing the noun classes in the randomly chosen examples"""
    set_a = set(a)
    set_b = set(b)
    if len(set_a.intersection(set_b)) > 0:
        return True
    else:
        return False


def gen_labels(clip1_feature_idxs, clip2_feature_idxs):
    """Generate the labels for each caption.
    Returns three lists of 20 elements - 0s and 1s"""
    label_caption1 = np.zeros(20)
    label_caption1[clip1_feature_idxs] = 1

    if len(clip2_feature_idxs) >= 1:
        label_caption2 = np.zeros(20)
        label_caption2[clip2_feature_idxs] = 1
    else:
        label_caption2 = np.ones(20)
        label_caption2 -= label_caption1

    label_caption3 = np.zeros(20)
    label_no_caption = np.ones(20)*0.5

    label_caption1 = torch.tensor(label_caption1, dtype=torch.float32)
    label_caption2 = torch.tensor(label_caption2, dtype=torch.float32)
    label_caption3 = torch.tensor(label_caption3, dtype=torch.float32)
    label_no_caption = torch.tensor(label_no_caption, dtype=torch.float32)

    return label_caption1, label_caption2, label_caption3, label_no_caption


def reformat_noun_classes(noun_classes):
    noun_classes = noun_classes.replace("['", "").replace("']", "")
    noun_classes = noun_classes.replace("'", "").replace("[", "").replace("]", "")
    noun_classes_reformatted = noun_classes.split(", ")
    return noun_classes_reformatted
