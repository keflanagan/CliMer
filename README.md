# CliMer
This repo contains code and data used in the BMVC paper 'Learning Temporal Sentence Grounding
From Narrated EgoVideos'.

## Features
The Omnivore visual features for the Ego4D and EPIC-Kitchens dataset splits can be found [here](insert dropbox link).

## Quick Start Guide
### Requirements
The dependencies can be found in Climer.yml
### Data
The 'data' folder contains csv files with the annotations for the train, validation and test splits for Ego4D and EPIC-Kitchens. It also contains metadata files which contain extra information used during training/testing.
The Omnivore visual features and the BERT features should be placed in the 'features' directory. Alternatively change the path to your chosen location for the features in the config file(s).


### Pretrained
Pretrained models can be found here (insert link)
### Training
To train the model, run the following command, replacing $DATASET with 'ego4d' for the Ego4D dataset and 'epic' for the EPIC-Kitchens dataset

```
bash train.sh $DATASET
```
Model checkpoints are saved by default in the 'checkpoints' directory.

### Evaluation
To evaluate the model, run the following command, again replacing $DATASET with the corresponding dataset (ego4d or epic)

```
bash test.sh $DATASET
```
