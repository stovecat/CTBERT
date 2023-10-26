# +
LD_LIBRARY_PATH=/usr/local/cuda/lib64
OMP_NUM_THREADS=9
export OMP_NUM_THREADS
LD_PRELOAD=/usr/local/lib/libjemalloc.so 

MODEL_NAME=microsoft/codebert-base
OUTPUT=../ckpt/pretrain/codeidiombert_only_python
TRAIN_FILE=../data/pretrain,train.pkl
EVAL_FILE=../data/pretrain,dev.pkl
MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
NODE_INDEX=0 && echo NODE_INDEX: ${NODE_INDEX}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=1 && echo NUM_NODE: ${NUM_NODE}
mkdir -p ${OUTPUT}
BLOCK_SIZE=512 # sentence length
TRAIN_BATCH_SIZE=16 # per gpu batch
EVAL_BATCH_SIZE=32
ACCUMULATE_STEPS=4
LANGUAGE=python
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
ADAM_EPS=1e-6
MAX_SREPS=20000
WARMUP_STEPS=2000
SAVE_STEPS=5000

# NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=ib0
cd ../src

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=${PER_NODE_GPU} --nnodes=${NUM_NODE} --node_rank=${NODE_INDEX} --master_port=2234 pretrain_code_idiom_python.py \
    --output_dir=$OUTPUT \
    --config_name=$MODEL_NAME \
    --model_name_or_path=$MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$EVAL_FILE \
    --lang $LANGUAGE \
    --block_size $BLOCK_SIZE \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATE_STEPS \
    --learning_rate $LEARNING_RATE \
    --node_index $NODE_INDEX \
    --gpu_per_node $PER_NODE_GPU \
    --weight_decay $WEIGHT_DECAY \
    --adam_epsilon $ADAM_EPS \
    --max_grad_norm 1.0 \
    --max_steps $MAX_SREPS \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --seed 123456 \
    --fp16 \
    --not_use_dfg \
    --use_code_idioms 2>&1| tee $OUTPUT/train.log
