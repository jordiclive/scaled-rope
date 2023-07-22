# Environment Configuration
source /p/home/jusers/clive1/juwels/clive1/miniconda3/bin/activate jordan_lora
export WANDB_API_KEY="d8216641d549f9bb3d0c5074baa39e15dfd55030"
export PYTHONPATH="/p/home/jusers/clive1/juwels/clive1/scaled-rope:$PYTHONPATH"
export TRANSFORMERS_CACHE="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export HF_DATASETS_CACHE="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export HF_HOME="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export PATH="/p/software/juwelsbooster/stages/2023/software/OpenMPI/4.1.4-GCC-11.3.0/bin:$PATH"


deepspeed --master_port 12802 \
          --launcher slurm \
          --hostfile '/p/home/jusers/clive1/juwels/hostfiles/hostfile.txt' \
          --master_addr $MASTER_ADDR \
          --no_ssh_check \
          /p/home/jusers/clive1/juwels/clive1/scaled-rope/finetune.py \
          --output_dir saved_ckpts_32k \
          --configs defaults lora-7b-llama2 \
          --deepspeed