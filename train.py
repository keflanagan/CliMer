#!/usr/bin/env python3
import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm
import pandas as pd
import argparse
from multiprocessing import cpu_count, set_start_method

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from args import create_parser
from lib.model.model_lightweight import TemporalGrounding
from lib.model.model_x import TemporalGroundingCross
from lib.dataset.dataset_ego4d_timestamp import Ego4dDatasetTimestamp
from lib.dataset.dataset_ego4d_full_video import Ego4dDatasetFullVideo
from lib.dataset.dataset_epic_timestamp import EpicDatasetTimestamp
from lib.dataset.dataset_epic_full_video import EpicDatasetFullVideo
from utils.data_utils import load_omnivore_clip_features, load_omnivore_clip_features_diff_videos, load_all_omnivore_features
from utils.utils import iou_stats, iou_time, iou_time_based, get_hard_preds, split_features, recombine_preds

sys.path.append('..')

try:
    set_start_method('spawn')
except RuntimeError:
    pass

# Define an argument parser to allow relevant parameters for training to be set
parser = create_parser()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
else:
    DEVICE = torch.device("cpu")


def main(args):
    # Create the train and test data loaders for loading data into the model
    if args.dataset == "ego4d":
        train_dataset = Ego4dDatasetTimestamp(args.caption_data_train, args.ego4d_metadata, args.bert_features,
                                              args.same_vid_sampling, args.combine, args.fixed_clip_length,
                                              args.clip_adjacent_timestamps,args.egovlp, args.egovlp_data,
                                              args.fps, args.feature_stride, DEVICE)

        test_dataset_full_video = Ego4dDatasetFullVideo(args.caption_data_val, args.bert_features,
                                                        args.same_vid_sampling, args.ego4d_metadata, args.fps,
                                                        args.feature_stride, DEVICE)
    else:
        train_dataset = EpicDatasetTimestamp(args.caption_data_train, args.epic_metadata, args.epic_all_data,
                                             args.bert_features, args.same_vid_sampling, args.combine,
                                             args.fixed_clip_length, args.clip_adjacent_timestamps, args.fps,
                                             args.feature_stride, DEVICE)

        test_dataset_full_video = EpicDatasetFullVideo(args.caption_data_val, args.bert_features,
                                                       args.same_vid_sampling, args.epic_metadata, args.fps,
                                                       args.feature_stride, DEVICE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    # VALIDATION
    test_loader_full_video = torch.utils.data.DataLoader(
        test_dataset_full_video,
        shuffle=False,
        batch_size=1,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    visual_feature_orig_dim = args.visual_feature_orig_dim  # omnivore feature size
    cap_orig_dim = args.cap_orig_dim
    visual_feature_input_dim = args.shared_projection_dim
    cap_embed_dim = args.shared_projection_dim
    feature_embed_dim = args.feature_embed_dim
    linear_hidden_dim = args.linear_hidden_dim
    num_heads = args.num_heads

    if args.cross_attention:
        model = TemporalGroundingCross(visual_projection_switch=True, visual_feature_orig_dim=visual_feature_orig_dim,
                                       visual_feature_input_dim=visual_feature_input_dim,
                                       feature_embed_dim=feature_embed_dim,
                                       linear_hidden_dim=linear_hidden_dim, num_heads=num_heads,
                                       cap_input_dim=cap_orig_dim, cap_embed_dim=cap_embed_dim,
                                       device=DEVICE)

    else:
        model = TemporalGrounding(visual_projection_switch=True, visual_feature_orig_dim=visual_feature_orig_dim,
                                  visual_feature_input_dim=visual_feature_input_dim,
                                  feature_embed_dim=feature_embed_dim, linear_hidden_dim=linear_hidden_dim,
                                  num_heads=num_heads, cap_input_dim=cap_orig_dim, cap_embed_dim=cap_embed_dim,
                                  device=DEVICE)

    # Define the loss function
    criterion_no_reduction = nn.BCELoss(reduction='none')  # criterion with no reduction for balanced losses
    criterion = nn.BCELoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Write training summary to the logs
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, test_loader_full_video,
        criterion, criterion_no_reduction, optimizer, summary_writer, args, DEVICE
    )

    trainer.train(
        args.epochs,
        args.log_frequency,
        check_frequency=args.checkpoint_frequency
    )

    summary_writer.close()


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader_full_video: DataLoader,
            criterion: nn.Module,
            criterion_no_reduction: nn.Module,
            optimizer: Optimizer,
            summary_writer: SummaryWriter,
            args,
            device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader_full_video = val_loader_full_video
        self.criterion = criterion
        self.criterion_no_reduction = criterion_no_reduction
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.args = args
        self.args = args
        self.step = 0
        # data loading
        self.omnivore_dict = load_all_omnivore_features(self.args.caption_data_train, self.args.caption_data_val,
                                                        self.args.caption_data_test, self.args.video_feature_path)

    def train(
            self,
            epochs: int,
            log_frequency: int = 5,
            start_epoch: int = 0,
            check_frequency: int = 1
    ):
        """Trains the model using the data returned from train_loader and saves
        the model weights

        Args:
            epochs:           The number of epochs for which to train the model
            log_frequency:    How often to log the training information in number of steps
            start_epoch:      Epoch number to start from
            check_frequency:  How often to save the model weights
        """

        # Step through the training for each batch within each epoch
        for epoch in tqdm(range(start_epoch, epochs)):
            total_loss = 0
            self.model.train()
            for combined_feature_idxs, bert_features, video_id, combined_video_ids, num_tokens, label_caption1, \
                    label_caption2, label_caption3, label_no_caption, captions, narrations in tqdm(self.train_loader):

                combined_video_ids = np.array(combined_video_ids).T.tolist()  # transpose as is transposed by dataloader

                if self.args.same_vid_sampling:
                    combined_features = load_omnivore_clip_features(self.omnivore_dict, combined_feature_idxs, video_id)
                else:
                    combined_features = load_omnivore_clip_features_diff_videos(self.omnivore_dict,
                                                                                combined_feature_idxs,
                                                                                combined_video_ids)
                combined_features = combined_features.detach()
                combined_features = combined_features.to(self.device)

                # caption embedding model
                bert_features_cap1 = bert_features[0].to(self.device)
                bert_features_cap2 = bert_features[1].to(self.device)
                bert_features_cap3 = bert_features[2].to(self.device)
                label_caption1 = label_caption1.to(self.device)
                label_caption2 = label_caption2.to(self.device)
                label_caption3 = label_caption3.to(self.device)
                label_no_caption = label_no_caption.to(self.device)
                num_tokens_cap1 = num_tokens[0].to(self.device)
                num_tokens_cap2 = num_tokens[1].to(self.device)
                num_tokens_cap3 = num_tokens[2].to(self.device)

                # Calculate the output logits
                preds_cap1 = self.model.predict(combined_features, bert_features_cap1, num_tokens_cap1, self.args.val,
                                                attenuation=True)
                preds_cap2 = self.model.predict(combined_features, bert_features_cap2, num_tokens_cap2, self.args.val)
                preds_cap3 = self.model.predict(combined_features, bert_features_cap3, num_tokens_cap3, self.args.val)
                preds_no_cap = self.model.predict(combined_features, bert_features_cap1, num_tokens_cap1, self.args.val,
                                                  attenuation=False)
                preds_cap1 = preds_cap1.squeeze(2)
                preds_cap2 = preds_cap2.squeeze(2)
                preds_cap3 = preds_cap3.squeeze(2)
                preds_no_cap = preds_no_cap.squeeze(2)

                if not self.args.combine:
                    loss_cap1 = self.get_balanced_loss_single_caption(label_caption1, preds_cap1)
                    loss_cap2 = torch.tensor([0])  # JUST A PLACEHOLDER
                    loss_cap3 = self.criterion(preds_cap3, label_caption3).type(torch.FloatTensor)
                    loss_no_cap = self.criterion(preds_no_cap, label_no_caption).type(torch.FloatTensor)

                else:
                    if self.args.balanced:
                        loss_cap1, loss_cap2 \
                            = self.get_balanced_loss_batch(label_caption1, label_caption2, preds_cap1, preds_cap2)

                    else:
                        loss_cap1 = self.criterion(preds_cap1, label_caption1).type(torch.FloatTensor)
                        loss_cap2 = self.criterion(preds_cap2, label_caption2).type(torch.FloatTensor)

                    loss_cap3 = self.criterion(preds_cap3, label_caption3).type(torch.FloatTensor)
                    loss_no_cap = self.criterion(preds_no_cap, label_no_caption).type(torch.FloatTensor)

                # GET TOTAL LOSS
                if self.args.combine:
                    combined_loss = torch.add(torch.add(torch.add(loss_cap1, loss_cap2), loss_cap3), loss_no_cap)
                else:
                    combined_loss = torch.add(torch.add(torch.mul(2, loss_cap1), loss_cap3), loss_no_cap)

                total_loss += combined_loss.item()

                # Perform the backward pass
                combined_loss.backward()

                # Step the optimizer and zero the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Print information about the training step
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, combined_loss)

                # VALIDATION
                if ((self.step + 1) % len(self.train_loader)) == 0:
                    self.validate_full_videos(epoch)
                    # Switch back to training mode
                    self.model.train()

                self.step += 1

            if (epoch + 1) % check_frequency == 0 or (epoch + 1) == epochs:
                checkpoint_path = args.checkpoint_path_root + '_ep' + str(epoch) + '.pth'
                print(f"Saving model to {checkpoint_path}")
                torch.save(self.model.state_dict(), checkpoint_path)

    def print_metrics(self, epoch, loss):
        epoch_step = self.step % len(self.train_loader)
        """Print out the training step information"""
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"data load time: "
        )

    def log_metrics(self, epoch, average_loss):
        """Add information to the logs"""
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "loss",
            {"train": average_loss},  # float(loss.item())},
            self.step
        )

    def validate_full_videos(self, epoch):
        iou_list = []
        preds1_list = []
        labels1_list = []
        captions_list = []
        feature_idxs_list = []
        feature_times_list = []
        clip_start_stop_time_list = []
        iou_time_list = torch.tensor([])

        self.model.eval()
        self.args.val = True
        for feature_idxs, feature_times, bert_features, num_tokens, label_caption1, caption1, clip_start_stop_time, \
                video_id in tqdm(self.val_loader_full_video):

            features = load_omnivore_clip_features(self.omnivore_dict, feature_idxs, video_id).to(DEVICE)

            bert_features_cap1 = bert_features.to(DEVICE)
            label_caption1 = label_caption1.to(DEVICE)
            num_tokens_cap1 = num_tokens.to(DEVICE)
            num_features = feature_idxs.shape[1]

            assert num_features == label_caption1.size()[1]

            # Calculate the output logits
            if num_features <= args.seg_size:
                preds_cap1 = self.model.predict(features, bert_features_cap1, num_tokens_cap1, self.args.val)
                preds_cap1 = preds_cap1.squeeze(2).detach()

            else:
                feature_segment_list, feature_idx_list = split_features(features, self.args.seg_size,
                                                                        self.args.overlap)
                full_pred_list = []
                for feature_segment in feature_segment_list:
                    preds_cap1 = self.model.predict(feature_segment, bert_features_cap1, num_tokens_cap1, self.args.val)
                    preds_cap1 = preds_cap1.squeeze(2).detach()
                    preds_cap1 = preds_cap1.tolist()
                    full_pred_list.append(preds_cap1)

                preds_cap1 = recombine_preds(full_pred_list, feature_idx_list, num_features)
                preds_cap1 = torch.tensor(preds_cap1).unsqueeze(0).to(DEVICE)
                preds_cap1 = preds_cap1.detach()

            # Now want to get hard predictions from these predictions:
            preds1_list.append(preds_cap1)
            labels1_list.append(label_caption1)
            captions_list.append(caption1)
            feature_idxs_list.append(feature_idxs)
            feature_times_list.append(feature_times)
            clip_start_stop_time_list.append(clip_start_stop_time)

            final_preds = get_hard_preds(preds_cap1, self.args.pred_threshold)
            iou_time = iou_time_based(final_preds, feature_times, clip_start_stop_time)
            iou_time_list = torch.cat((iou_time_list, iou_time))
            feature_times_list.append(feature_times)
            clip_start_stop_time_list.append(clip_start_stop_time)

        self.args.val = False

        # iou stats

        iou_time_above_1, iou_time_above_3, iou_time_above_5, iou_time_above_7, \
            frac_time_above_1, frac_time_above_3, frac_time_above_5, frac_time_above_7 \
            = iou_stats(iou_time_list)

        mr = (frac_time_above_1 + frac_time_above_3 + frac_time_above_5) / 3

        with open(args.results_path + 'test_clean.txt', 'a') as file:
            file.write(f'Epoch {epoch} Threshold {self.args.pred_threshold}  0.1: {round(frac_time_above_1, 3)}, '  
                       f'0.3: {round(frac_time_above_3, 3)}, 0.5: {round(frac_time_above_5, 3)}, '
                       f'0.7: {round(frac_time_above_7, 3)}, mR: {round(mr, 3)} \n')

        print(f'Epoch {epoch} Threshold {self.args.pred_threshold}  0.1: {round(frac_time_above_1, 3)}, '  
              f'0.3: {round(frac_time_above_3, 3)}, 0.5: {round(frac_time_above_5, 3)}, '
              f'0.7: {round(frac_time_above_7, 3)}, mR: {round(mr, 3)}')

        return iou_list, captions_list, preds1_list, labels1_list, feature_idxs_list, feature_times_list, \
            clip_start_stop_time_list

    def get_balanced_loss_batch(self, label_caption1, label_caption2, preds_cap1, preds_cap2):
        # positive
        weights_cap1 = torch.zeros(label_caption1.size()).to(self.device)
        preds_cap1_pos_loc = torch.nonzero(label_caption1)
        preds_cap1_neg_loc = (label_caption1 == 0).nonzero().squeeze(1)

        weights_cap2 = torch.zeros(label_caption2.size()).to(self.device)
        preds_cap2_pos_loc = torch.nonzero(label_caption2)
        preds_cap2_neg_loc = (label_caption2 == 0).nonzero().squeeze(1)

        loss_size_retention_factor_cap1 = []
        loss_size_retention_factor_cap2 = []

        for i in range(label_caption1.size()[0]):
            weights_cap1, loss_size_retention_factor_cap1_part = \
                self.get_weights_single_caption(preds_cap1_pos_loc, preds_cap1_neg_loc, weights_cap1, label_caption1,
                                                preds_cap1, i)

            weights_cap2, loss_size_retention_factor_cap2_part = \
                self.get_weights_single_caption(preds_cap2_pos_loc, preds_cap2_neg_loc, weights_cap2, label_caption2,
                                                preds_cap2, i)

            loss_size_retention_factor_cap1.append(loss_size_retention_factor_cap1_part)
            loss_size_retention_factor_cap2.append(loss_size_retention_factor_cap2_part)

        loss_size_retention_factor_cap1 = torch.tensor(loss_size_retention_factor_cap1).unsqueeze(1).to(self.device)
        loss_size_retention_factor_cap2 = torch.tensor(loss_size_retention_factor_cap2).unsqueeze(1).to(self.device)
        individual_loss_cap1 = self.criterion_no_reduction(preds_cap1, label_caption1)

        loss_cap1 = torch.mean(
            torch.mul(torch.mul(weights_cap1, individual_loss_cap1), loss_size_retention_factor_cap1))
        individual_loss_cap2 = self.criterion_no_reduction(preds_cap2, label_caption2)
        loss_cap2 = torch.mean(
            torch.mul(torch.mul(weights_cap2, individual_loss_cap2), loss_size_retention_factor_cap2))

        return loss_cap1, loss_cap2

    def get_balanced_loss_single_caption(self, label_caption1, preds_cap1):

        weights_cap1 = torch.zeros(label_caption1.size()).to(self.device)
        preds_cap1_pos_loc = torch.nonzero(label_caption1)
        preds_cap1_neg_loc = (label_caption1 == 0).nonzero().squeeze(1)

        loss_size_retention_factor_cap1 = []

        for i in range(label_caption1.size()[0]):
            weights_cap1, loss_size_retention_factor_cap1_part = \
                self.get_weights_single_caption(preds_cap1_pos_loc, preds_cap1_neg_loc, weights_cap1, label_caption1,
                                                preds_cap1, i)

            loss_size_retention_factor_cap1.append(loss_size_retention_factor_cap1_part)

        loss_size_retention_factor_cap1 = torch.tensor(loss_size_retention_factor_cap1).unsqueeze(1).to(self.device)
        individual_loss_cap1 = self.criterion_no_reduction(preds_cap1, label_caption1)
        loss_cap1 = torch.mean(
            torch.mul(torch.mul(weights_cap1, individual_loss_cap1), loss_size_retention_factor_cap1))

        return loss_cap1

    @staticmethod
    def get_weights_single_caption(preds_cap_pos_loc, preds_cap_neg_loc, weights_cap, label_caption, preds_cap,
                                         i):
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


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        "train_"
        f"bs={args.train_batch_size}_"
        f"lr={args.learning_rate}_"
        f"run_"
)
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
