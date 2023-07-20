# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import math
import torch.nn as nn
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import argparse
from datetime import datetime
import os
import gzip
import csv
from datasets import load_dataset
from torch import cuda
import pandas as pd




def train_mpnet(dataset):
    language = dataset.split('/')[1]
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    train_batch_size = 128
    num_epochs = 5
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    max_seq_length = 32
    out_features = 128 # 自定义bert输出维度
    model_save_path = 'mpnet_output/' + dataset.split('/')[0] + '_' + dataset.split('/')[1]# + '_' + str(out_features)



    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    # dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features, activation_function=nn.Tanh())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device='cuda:0') # 输出自定义维度
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda:0') # 输出768维度

    """For SimCSE, we create our train_samples with InputExamples that consists of two times the same sentences, i.e.
    ```
    train_samples = [InputExample(texts=["sent1", "sent1"]), InputExample(texts=["sent2", "sent2"]), ...]
    ````
    """
    # fine-tune
    # random dropout 构造正例和反例
    train_samples = []
    if 'DBP15k' in dataset:
        text_set1 = dataset + '/id2trans_'+ language.split('_')[0] + '.txt'
        with open(text_set1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                text = line.split('\t')[1]
                train_samples.append(InputExample(texts=[text,text]))

        text_set2 = dataset + '/id2entity_en.txt'
        with open(text_set2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                text = line.split('\t')[1]
                train_samples.append(InputExample(texts=[text,text]))
    elif 'DWY100k' in dataset:
        text_set1 = dataset + '/id2entity_' + language.split('_')[0] + '.txt'
        with open(text_set1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                text = line.split('\t')[1]
                train_samples.append(InputExample(texts=[text, text]))

        text_set2 = dataset + '/id2entity_' + language.split('_')[1] + '.txt'
        with open(text_set2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                text = line.split('\t')[1]
                train_samples.append(InputExample(texts=[text, text]))
    """logging.info("Read Amazon training dataset")
    with open('Amazon_sports_with_attr.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_samples.append(InputExample(texts=[row['title'], row['title']]))
    
    logging.info("Read Ebay training dataset")
    with open('eBay_sports_en.csv') as csvfile_ebay:
        reader = csv.DictReader(csvfile_ebay)
        for row in reader:
            train_samples.append(InputExample(texts=[row['name'], row['name']]))"""

    logging.info("{} train sentences".format(len(train_samples)))

    """As loss, we use: MultipleNegativesRankingLoss
    Here, texts[0] and texts[1] are considered as positive pair, while all others are negatives in a batch
    """

    # Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []
    dev_dataset_en = load_dataset("stsb_multi_mt", name="en", split="dev") # 用stsb验证embedding是否准确
    dev_dataset_zh = load_dataset("stsb_multi_mt", name="zh", split="dev")

    for row in dev_dataset_en:
        score = float(row['similarity_score']) / 5.0  # Normalize score to range 0 ... 1
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    for row in dev_dataset_zh:
        score = float(row['similarity_score']) / 5.0  # Normalize score to range 0 ... 1
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    """logging.info("read my dataset")
    dev_samples = []
    dev_dataset = pd.read_csv('fake_sign.csv',encoding='gbk')
    
    for ind in dev_dataset.index:
        score = dev_dataset['score'][ind] / 10
        dev_samples.append(InputExample(texts=[dev_dataset['sentence1'][ind], dev_dataset['sentence2'][ind]], label=score))
    logging.info("{} dev sentences".format(len(dev_dataset)))"""

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

    # Use MultipleNegativesRankingLoss for SimCSE
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Performance before training")
    dev_evaluator(model)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=100,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    # model = SentenceTransformer(model_save_path)

if __name__ == '__main__':
    '''
    python sbert_simcse.py --dataset DBP15k/zh_en
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DWY100k/yg_dbp')  # , required=True) DWY100k/wd_dbp, DBP15k/zh_en
    args = parser.parse_args()
    train_mpnet(args.dataset)
