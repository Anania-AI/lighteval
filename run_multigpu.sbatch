#!/bin/bash
#SBATCH --job-name=qwen-3-4b-it_eval
#SBATCH --account=project_465002461   # Your specific LUMI project billing account
#SBATCH --partition=standard-g        # The LUMI partition for standard GPU jobs
#SBATCH --nodes=1                  # <--- Request 2 nodes
#SBATCH --gpus-per-node=1          # 8 GPUs per node
#SBATCH --ntasks-per-node=1        # 1 torchrun process per node
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
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
# export SINGULARITYENV_ROCR_VISIBLE_DEVICES=0,1
# export SINGULARITYENV_HIP_VISIBLE_DEVICES=0,1

# export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# export SINGULARITYENV_NCCL_NET_GDR_LEVEL=3
# Force each GPU to use a unique, local high-speed cache instead of the network drive

export SINGULARITYENV_MIOPEN_USER_DB_PATH="/tmp/miopen_db_$SLURM_JOB_ID_$SLURM_PROCID"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen_cache_$SLURM_JOB_ID_$SLURM_PROCID"

ALL_TASKS="armenian:mmlu_pro|0,armenian:exam_math|0,armenian:exam_history|0,armenian:exam_literature|0,armenian:topic-14class|0,armenian:sentiment|0,armenian:space_fix|0,armenian:punctuation|0,armenian:finer|0,armenian:pioner|0,armenian:pos|0,armenian:arak|0,armenian:ms_marco|0,armenian:squad|0,armenian:belebele|0,armenian:scientific|0,armenian:syndarin|0,armenian:dream|0,armenian:include|0,armenian:hartak|0,armenian:email|0,armenian:conversation|0,armenian:paraphrase|0,armenian:short_sentences_translation|0"
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

srun --cpu-bind=none \
    ../new_container_exec_torch_2.7.1.sh \
    ../new_setup_env_lighteval.sh lighteval \
    lighteval vllm \
    "model_name=$MODEL_NAME,dtype=bfloat16,max_model_length=4096" \
    "$ALL_TASKS" \
    --custom-tasks src/lighteval/tasks/multilingual/tasks/armenian.py \
    --output-dir "evaluations/$MODEL_NAME"

