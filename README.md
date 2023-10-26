# CliMer
This repo contains code and data used in the BMVC paper 'Learning Temporal Sentence Grounding
From Narrated EgoVideos'.

## Features
The Omnivore visual features and BERT text features for the Ego4D and EPIC-Kitchens dataset splits will be released soon.

## Quick Start Guide
### Requirements
The dependencies can be found in Climer.yml

This is a conda environment and can be installed with the following command:

```
conda env create -f climer.yml
```
### Ego4D/EPIC-Kitchens Dataset Splits
The 'data' folder contains csv files with the annotations for the train, validation and test splits for Ego4D and EPIC-Kitchens. It also contains metadata files which contain extra information used during training/testing.

### Features
The Omnivore visual features and the BERT features should be placed in the 'features' directory. Alternatively change the path to your chosen location for the features in the config file(s).


### Pretrained
Pretrained models will be released soon.
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
