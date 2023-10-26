#!/bin/bash
LD_PRELOAD=/usr/local/lib/libjemalloc.so 
cuda=$1
lang=python
path=../ckpt/codesearchnet/graphcodebert_$lang
data_path=../data/adv_test
mkdir -p $path
cd ../src
CUDA_VISIBLE_DEVICES=$cuda python inference_codeidiombert_python_adv_test.py \
    --output_dir=$path \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --lang=$lang \
    --train_data_file=$data_path/train.jsonl \
    --eval_data_file=$data_path/valid.jsonl \
    --test_data_file=$data_path/test.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --data_flow_length 64 \
    --idiom_length 0 \
    --train_batch_size 32 \
    --eval_batch_size 256 \
    --learning_rate 2e-5 \
    --do_test \
    --raw_code_key function \
    --seed 123456 2>&1| tee $path/adv_test.log
