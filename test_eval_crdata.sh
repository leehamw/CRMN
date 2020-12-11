#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath_test='python3'
pythonpath_eval='python3 -m'

# set parameters
data_name=crdata
mode=valid  # only valid
data_dir=./data/CRDATA
save_dir=./models-crdata-v3
output_dir=./outputs-crdata-v3
ckpt=best.model
beam_size=2
max_dec_len=65

mkdir -p ${output_dir}/${ckpt}/${mode}

${pythonpath_test} ./main.py --test --mode=${mode} --data_dir=${data_dir} --save_dir=${save_dir} --ckpt=${ckpt} --beam_size=${beam_size} --max_dec_len=${max_dec_len} --save_file=${output_dir}/${ckpt}/${mode}/output.txt
${pythonpath_eval} tools.eval --mode=${mode} --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}/${mode}

for i in {1..15}
do
    ckpt=state_epoch_${i}.model
    mkdir -p ${output_dir}/${ckpt}/${mode}

    ${pythonpath_test} ./main.py --test --mode=${mode} --data_dir=${data_dir} --save_dir=${save_dir} --ckpt=${ckpt} --beam_size=${beam_size} --max_dec_len=${max_dec_len} --save_file=${output_dir}/${ckpt}/${mode}/output.txt
    ${pythonpath_eval} tools.eval --mode=${mode} --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}/${mode}
done
