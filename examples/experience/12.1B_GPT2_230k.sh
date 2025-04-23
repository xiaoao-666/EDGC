#! /bin/bash

# Runs the "13B" parameter model

NODE_DIR="../node"
NODE_RANK_FILE="$NODE_DIR/node_rank_file_2.txt"
GPUS_PER_NODE=4
NNODES=16
MASTER_PORT=6000
ip_address=$(hostname -I | awk '{print $1}') 
first_ip=$(head -n 1 "$NODE_RANK_FILE" | awk -F: '{print $1}')
current_node_rank=$(grep -oP "^$ip_address:\K\d+" "$NODE_RANK_FILE") 
NODE_RANK=$current_node_rank
echo "Current node rank: $current_node_rank"
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/data/gpt2_openwebtext
CHECKPOINT_PATH=/checkpoint/13B_dygc_230k/

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $first_ip --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../../pretrain_gpt.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \
       --num-layers 76 \
       --hidden-size 3584 \
       --num-attention-heads 28 \
       --micro-batch-size 4 \
       --global-batch-size 128 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 230000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --collect-log-path /iteration_log/12.1B_edgc_230k \
       --tensorboard-dir /tensorboard/12.1B_edgc_230k \
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
