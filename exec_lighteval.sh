#!/bin/bash

GPUS_PER_NODE=2  # Keeping it simple for a toy run
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -i 25000-35000 -n 1)
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export WORLD_SIZE=$GPUS_PER_NODE
export RANK=$NODE_RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT



ALL_TASKS="mmlu:miscellaneous|5,arc:challenge|0,gsm8k|5,mmlu_pro|5,armenian:topic-14class|0,armenian:sentiment|0,armenian:space_fix|0,armenian:punctuation|0,armenian:finer|0,armenian:pioner|0,armenian:pos|0,armenian:arak|0,armenian:ms_marco|0,armenian:squad|0,armenian:belebele|0,armenian:scientific|0,armenian:syndarin|0,armenian:dream|0,armenian:include|0,armenian:hartak|0,armenian:email|0,armenian:conversation|0,armenian:paraphrase|0,armenian:short_sentences_translation|0"


../new_container_exec_torch_2.7.1.sh ../new_setup_env.sh \
lighteval accelerate \
    "model_name=Metric-AI/ArmLlama-3.2-1B-PT-11200,batch_size=1" \
    "$ALL_TASKS" \
    --custom-tasks src/lighteval/tasks/multilingual/tasks/armenian.py \
    --max-samples 1
