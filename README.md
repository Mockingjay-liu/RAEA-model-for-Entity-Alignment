## Introduction
This is the Pytorch implementation for our paper at World Wide Web 2023: **Cross-platform productmatching based on entity alignment of knowledge graph with raea model**. https://doi.org/10.1007/s11280-022-01134-y

## Run RAEA
First train the RAEA model on the DBP15k/zh_en dataset with default hyperparameters: `python train_subgraph.py`. The trained models of defferent channels will be saved.

Then, ensemble the channels and valid the performance on the test set: `python ensemble_subgraphs.py`

The hyperparameters we use are set to the default value.

You can change the value of hyperparameters. For example, the command for training `python train_subgraph.py --dataset DBP15k/zh_en --gpu_id 0 --nega_sample_num 15` defines the `dataset`, `gup_id` and `nega_sample_num` 

To understand the hyperparameters setting, please refer to "Training Details" part of our paper.

## Citation
Please kindly cite our work as follows if you find our paper or codes helpful.

Liu W, Pan J, Zhang X, et al. Cross-platform product matching based on entity alignment of knowledge graph with raea
model. World Wide Web (2023). https://doi.org/10.1007/s11280-022-01134-y
