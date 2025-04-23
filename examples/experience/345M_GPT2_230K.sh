#! /bin/bash

# Runs the "345M" parameter model

NODE_DIR="../node"
NODE_RANK_FILE="$NODE_DIR/node_rank_file_3.txt"
GPUS_PER_NODE=2
NNODES=8
MASTER_PORT=6000
ip_address=$(hostname -I | awk '{print $1}')
first_ip=$(head -n 1 "$NODE_RANK_FILE" | awk -F: '{print $1}')
current_node_rank=$(grep -oP "^$ip_address:\K\d+" "$NODE_RANK_FILE")
NODE_RANK=$current_node_rank
echo "Current node rank: $current_node_rank"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/data/my-gpt2_text_document
CHECKPOINT_PATH=/checkpoint/345M_edgc_230k

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $first_ip --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../../pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 230000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --collect-log-path /iteration_log/test \
       --tensorboard-dir /tensorboard/test \
       --data-path $DATA_PATH \
       --vocab-file /data/gpt2-vocab.json \
       --merge-file /data/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --override-opt_param-scheduler \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --grad_comp \
       --use_error_feedback \
       --DDP-impl local 
