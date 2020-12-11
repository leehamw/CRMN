#!/bin/bash

# set python path
pythonpath='python3 -m'

# set dir
mode=test  # only test
data_name='crdata'
data_dir=./data/CRDATA
eval_dir=./outputs-crdata-v3/best.model

${pythonpath} tools.submit --mode=${mode} --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${eval_dir}/${mode}