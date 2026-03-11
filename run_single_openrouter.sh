ALL_TASKS="armenian:mmlu_pro|0,armenian:exam_math|0,armenian:exam_history|0,armenian:exam_literature|0,armenian:topic-14class|0,armenian:sentiment|0,armenian:space_fix|0,armenian:punctuation|0,armenian:finer|0,armenian:pioner|0,armenian:pos|0,armenian:arak|0,armenian:ms_marco|0,armenian:squad|0,armenian:belebele|0,armenian:scientific|0,armenian:syndarin|0,armenian:dream|0,armenian:include|0,armenian:hartak|0,armenian:email|0,armenian:conversation|0,armenian:paraphrase|0,armenian:short_sentences_translation|0"
# ALL_TASKS="armenian:exam_math|0"

MODEL_NAME="model_name=openrouter/qwen/qwen3.5-35b-a3b,max_model_length=4092"
SHORT_NAME=$(basename "$MODEL_NAME")

export OPENROUTER_API_KEY=""

export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

lighteval endpoint litellm \
    "$MODEL_NAME" \
    "$ALL_TASKS" \
    --custom-tasks src/lighteval/tasks/multilingual/tasks/armenian.py \
    --output-dir "evaluations" \
    --results-path-template "evaluations/$SHORT_NAME" \
    --save-details 
