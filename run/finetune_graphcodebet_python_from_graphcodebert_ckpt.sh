LD_PRELOAD=/usr/local/lib/libjemalloc.so 
lang=$1
cuda=$2
path=../ckpt/codesearchnet/graphcodebert_python_from_graphcodebert_ckpt_40k_$lang
init_path=../ckpt/pretrain/graphcodebert_python_from_graphcodebert_ckpt/checkpoint-last
data_path=../data/codesearchnet
mkdir -p $path
cd ../src
CUDA_VISIBLE_DEVICES=$cuda python finetune_codeidiombert_python.py \
    --output_dir=$path \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=$init_path \
    --tokenizer_name=microsoft/codebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=$data_path/$lang/train.jsonl \
    --eval_data_file=$data_path/$lang/valid.jsonl \
    --test_data_file=$data_path/$lang/test.jsonl \
    --codebase_file=$data_path/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --data_flow_length 64 \
    --idiom_length 0 \
    --train_batch_size 32 \
    --eval_batch_size 256 \
    --learning_rate 2e-5 \
    --do_test \
    --seed 123456 2>&1| tee $path/train.log
