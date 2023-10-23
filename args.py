import argparse
from pathlib import Path

def create_parser():
    parser = argparse.ArgumentParser(
        description="Train a temporal grounding model on EPIC-Kitchens data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="ego4d", type=str, help="which dataset to use, ego4d or epic")
    parser.add_argument("--val", default=False)
    parser.add_argument("--model-path", default="/media/viki/DATA/Kevin/temp_grounding/epic/checkpoints/ego4d_omnivore_newBERT_cooking_crossattention_fullloopresidual_clip_posenc_vidsep_val_proj2048_htc_sv_bs32_lr10e5_100epochs_nonrepeatedcaptions_timestamp50100_checkpoint_ep13.pth")
    parser.add_argument("--video-feature-path", default=f"/media/barry/DATA/Kevin/ego4d/omnivore_clip_cooking/")
    parser.add_argument("--caption-data-train", default=f"/media/barry/DATA/Kevin/ego4d/annotations/ego4d_train_v2_inc_new_test_videos_verb_noun_cooking.csv")
    parser.add_argument("--caption-data-val", default=f"/media/barry/DATA/Kevin/ego4d/annotations/ego4d_VAL_v2_non_repeat_vn_inc_new_test_videos_cooking_nobike.csv")
    parser.add_argument("--caption-data-test", default=f"/media/barry/DATA/Kevin/ego4d/annotations/ego4d_VAL_v2_non_repeat_vn_inc_new_test_videos_cooking_nobike.csv")
    parser.add_argument("--ego4d-metadata", default=f"/media/barry/DATA/Kevin/ego4d/annotations/ego4d_metadata.csv")
    parser.add_argument("--epic-metadata", default="/media/barry/DATA/Kevin/epic/annotations/EPIC_100_video_info.csv")
    parser.add_argument("--epic-all-data", default=f"/media/barry/DATA/Kevin/epic/annotations/EPIC_100_train_adjusted.csv")
    parser.add_argument("--bert-features", default=f"/media/barry/DATA/Kevin/ego4d/BERT_cap_features_ego4d_real_cooking.lmdb")
    parser.add_argument("--results-path", default=f"/media/eve/DATA/Kevin/temp_grounding/ego4d/val_results/final_model/")
    parser.add_argument("--egovlp-data", default=f"/media/barry/DATA/Kevin/ego4d/annotations/egovlp_params_correct.json")
    parser.add_argument("--random-baseline", default=False, type=bool, help="Whether to use random baseline for evaluation")
    parser.add_argument("--log-dir", default=Path(f"/media/barry/DATA/Kevin/temp_grounding/ego4d/logs"), type=Path)
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--train-batch-size", default=32, type=int, help="Number of examples in each batch")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train for")
    parser.add_argument("--val-frequency", default=1, type=int, help="How frequently to test the model on the validation set in number of epochs")
    parser.add_argument("--log-frequency", default=10, type=int, help="How frequently to save logs to tensorboard in number of steps")
    parser.add_argument("--checkpoint-save-path", type=Path, default='/media/eve/DATA/Kevin/temp_grounding/ego4d/checkpoints/test_clean')
    parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs")
    parser.add_argument("--worker-count", default=6, type=int, help="Number of worker processes used to load data.")
    parser.add_argument("--same-vid-sampling", default=True, type=bool, help="whether or not combined videos should be sampled from the same video")
    parser.add_argument("--cross-attention", default=True, type=bool, help="Whether to use cross attention between words and frames as interaction rather than hadamard product")
    parser.add_argument("--balanced", default=True, type=bool, help="Whether to balance positive and negative frames in the loss")
    parser.add_argument("--combine", default=True, type=bool, help="Whether to use the video combination trick or to just use a single caption in its own sequence")
    parser.add_argument("--fixed-clip-length", default="None", type=str, help="Whether to use a fixed clip length - value is string of the length or 'None'")
    parser.add_argument("--clip-adjacent-timestamps", default="None", type=str, help="Whether to use adjacent timestamps as clip boundaries - value is 'half', 'full' or 'None'")
    parser.add_argument("--egovlp", default=False, type=bool, help="Whether to use egovlp method for generating clips")
    parser.add_argument("--fps", default=30, type=float, help="FPS of the videos for which features have been generated")
    parser.add_argument("--feature-stride", default=16, type=float, help="stride in number of frames of the features generated for each video")
    parser.add_argument("--seg-size", default=2000, type=float, help="Segment size for inference")
    parser.add_argument("--overlap", default=1000, type=float, help="Overlap between segments for inference")
    parser.add_argument("--pred-threshold", default=0.8, type=float, help="Prediction threshold during evaluation")
    parser.add_argument("--visual-feature-orig-dim", default=1536, type=int, help="size of input visual features")
    parser.add_argument("--cap-orig-dim", default=768, type=int, help="size of input caption features")
    parser.add_argument("--shared-projection-dim", default=2048, type=int, help="size of projection")
    parser.add_argument("--feature-embed-dim", default=3072, type=int, help="size of feature embedding dimension, split across heads")
    parser.add_argument("--linear-hidden-dim", default=1024, type=int, help="size of hidden dimension in linear layers")
    parser.add_argument("--num-heads", default=6, type=int, help="size of projection")
    
    return parser


