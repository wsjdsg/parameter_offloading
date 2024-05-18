#!/bin/bash
OPTS=""
OPTS+=" --model-config 1b"
OPTS+=" --batch-size 1"
OPTS+=" --train-iters 1400"
OPTS+=" --max-length 1024"
OPTS+=" --lr 1e-5"
OPTS+=" --weight-decay 1e-2" 
OPTS+=" --clip-grad 1.0" 
OPTS+=" --epochs 1"
OPTS+=" --offloading" #开启offloading
OPTS+=' --fp16'   #使用fp16
OPTS+=" $@"

python train.py ${OPTS} 