ALL_TASKS="mmlu:miscellaneous|5,arc:challenge|0,gsm8k|5,mmlu_pro|5,armenian:topic-14class|0,armenian:sentiment|0,armenian:space_fix|0,armenian:punctuation|0,armenian:finer|0,armenian:pioner|0,armenian:pos|0,armenian:arak|0,armenian:ms_marco|0,armenian:squad|0,armenian:belebele|0,armenian:scientific|0,armenian:syndarin|0,armenian:dream|0,armenian:include|0,armenian:hartak|0,armenian:email|0,armenian:conversation|0,armenian:paraphrase|0,armenian:short_sentences_translation|0"

lighteval accelerate \
    "model_name=Metric-AI/ArmLlama-3.2-1B-PT-11200,batch_size=1,device=cpu" \
    "$ALL_TASKS" \
    --custom-tasks src/lighteval/tasks/multilingual/tasks/armenian.py \
    --max-samples 1
    # armenian:topic-14class \
