#!/bin/bash
#SBATCH --job-name=gemini_3_pro_eval
#SBATCH --account=project_465002461   # Your specific LUMI project billing account
#SBATCH --partition=small        # The LUMI partition for standard GPU jobs
#SBATCH --nodes=1                  # <--- Request 2 nodes
#SBATCH --gpus-per-node=0          # 8 GPUs per node
#SBATCH --ntasks-per-node=1        # 1 torchrun process per node
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/pfs/lustref1/flash/project_465002461/LLM/Spartak/lighteval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master Node IP: $head_node_ip"

export SINGULARITYENV_FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
export PYTHONPATH=$PYTHONPATH:/pfs/lustref1/flash/project_465002461/LLM/Spartak/lighteval/src
export SINGULARITYENV_PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH

export SINGULARITYENV_MIOPEN_USER_DB_PATH="/tmp/miopen_db_$SLURM_JOB_ID_$SLURM_PROCID"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen_cache_$SLURM_JOB_ID_$SLURM_PROCID"

# ALL_TASKS="armenian:mmlu_pro|0,armenian:exam_math|0,armenian:exam_history|0,armenian:exam_literature|0,armenian:topic-14class|0,armenian:sentiment|0,armenian:space_fix|0,armenian:punctuation|0,armenian:finer|0,armenian:pioner|0,armenian:pos|0,armenian:arak|0,armenian:ms_marco|0,armenian:squad|0,armenian:belebele|0,armenian:scientific|0,armenian:syndarin|0,armenian:dream|0,armenian:include|0,armenian:hartak|0,armenian:email|0,armenian:conversation|0,armenian:paraphrase|0,armenian:short_sentences_translation|0"
ALL_TASKS="armenian:mmlu_pro|0"
MODEL_NAME="openrouter/google/gemini-3-pro-preview"

export OPENROUTER_API_KEY="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"
export SINGULARITYENV_OPENROUTER_API_KEY="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"

export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export SINGULARITYENV_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

export SINGULARITYENV_http_proxy=$http_proxy
export SINGULARITYENV_https_proxy=$https_proxy
export SINGULARITYENV_HTTP_PROXY=$HTTP_PROXY
export SINGULARITYENV_HTTPS_PROXY=$HTTPS_PROXY

# Pass the no_proxy variables so internal LUMI traffic doesn't get routed externally
export SINGULARITYENV_no_proxy=$no_proxy
export SINGULARITYENV_NO_PROXY=$NO_PROXY

export SINGULARITYENV_LITELLM_LOG="DEBUG"
export SINGULARITYENV_HF_HUB_VERBOSITY="debug"

srun --cpu-bind=none \
    ../new_container_exec_torch_2.7.1.sh \
    ../new_setup_env_lighteval.sh lighteval \
    lighteval endpoint litellm \
    "$MODEL_NAME" \
    "$ALL_TASKS" \
    --custom-tasks src/lighteval/tasks/multilingual/tasks/armenian.py \
    --output-dir "evaluations/$MODEL_NAME"
