#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath='python3'

# set parameters
mode=test  # test or valid
data_dir=./data/CRDATA
save_dir=./models-crdata-v3
output_dir=./outputs-crdata-v3
ckpt=best.model
beam_size=2
max_dec_len=65  # vaild:65 test:50

mkdir -p ${output_dir}/${ckpt}/${mode}

${pythonpath} ./main.py --test --mode=${mode} --data_dir=${data_dir} --save_dir=${save_dir} --ckpt=${ckpt} --beam_size=${beam_size} --max_dec_len=${max_dec_len} --save_file=${output_dir}/${ckpt}/${mode}/output.txt
