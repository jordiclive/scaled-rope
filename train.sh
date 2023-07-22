# Environment Configuration
source /p/home/jusers/clive1/juwels/clive1/miniconda3/bin/activate jordan_lora
export WANDB_API_KEY="d8216641d549f9bb3d0c5074baa39e15dfd55030"
export PYTHONPATH="/p/home/jusers/clive1/juwels/clive1/scaled-rope:$PYTHONPATH"
export TRANSFORMERS_CACHE="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export HF_DATASETS_CACHE="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export HF_HOME="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export PATH="/p/software/juwelsbooster/stages/2023/software/OpenMPI/4.1.4-GCC-11.3.0/bin:$PATH"
export HUGGING_FACE_HUB_TOKEN="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"


deepspeed /p/home/jusers/clive1/juwels/clive1/scaled-rope/finetune.py \
          --output_dir saved_ckpts_32k \
          --configs defaults lora-7b-llama2 \
          --deepspeed