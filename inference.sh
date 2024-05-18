#!/bin/bash
OPTS=""
OPTS+=" --model-config 1b"
OPTS+=" --batch-size 1"
OPTS+=" --train-iters 1400"
OPTS+=" --max-length 512"
OPTS+=" --offloading" #开启offloading
OPTS+=" $@"

python inference.py ${OPTS}  