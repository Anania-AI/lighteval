export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/pfs/lustref1/flash/project_465002461/LLM/Spartak/lighteval/src

export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

export VLLM_WORKER_MULTIPROC_METHOD=spawn

ray stop
ray start --head --num-gpus=2

ALL_TASKS="armenian:exam_history|0"
# ALL_TASKS="armenian:mmlu_pro|0,armenian:exam_math|0,armenian:exam_history|0,armenian:exam_literature|0,armenian:topic-14class|0,armenian:sentiment|0,armenian:space_fix|0,armenian:punctuation|0,armenian:finer|0,armenian:pioner|0,armenian:pos|0,armenian:arak|0,armenian:ms_marco|0,armenian:squad|0,armenian:belebele|0,armenian:scientific|0,armenian:syndarin|0,armenian:dream|0,armenian:include|0,armenian:hartak|0,armenian:email|0,armenian:conversation|0,armenian:paraphrase|0,armenian:short_sentences_translation|0"

lighteval vllm \
    "model_name=Metric-AI/ArmLlama-3.2-1B-PT-11200,dtype=float16,data_parallel_size=2" \
    "$ALL_TASKS" \
    --custom-tasks src/lighteval/tasks/multilingual/tasks/armenian.py \
    --max-samples 1
