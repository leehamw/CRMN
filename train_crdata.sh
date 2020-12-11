#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath='python3'

# set parameters
data_dir=./data/CRDATA
save_dir=./models-crdata-v3
ckpt=
num_epochs=15
pre_epochs=15
embed_file=./data/CRDATA/sgns.weibo.bigram-char  # todo control whether to load pre_trained embeddings in Field
train_embed=True  # control whether to update pre_trained embeddings if used when training
use_embed=False  # control whether to load pre_trained embeddings from Field to Embedder

${pythonpath} ./main.py --data_dir=${data_dir} --ckpt=${ckpt} --save_dir=${save_dir} --num_epochs=${num_epochs} --pre_epochs=${pre_epochs} --embed_file=${embed_file} --train_embed=${train_embed} --use_embed=${use_embed}
