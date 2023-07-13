conda activate /mnt/data/jordiclive/jordan_scaled

export HOME="/mnt/data/jordiclive"
export TMP="/mnt/data/jordiclive"
export TEMP="/mnt/data/jordiclive"
export TMPDIR="/mnt/data/jordiclive"
export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_DATASETS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_HOME="/mnt/data/jordiclive/transformers_cache"
export WANDB_API_KEY="d8216641d549f9bb3d0c5074baa39e15dfd55030"


export PYTHONPATH="/mnt/data/jordiclive/scaled-rope:$PYTHONPATH"

cd /mnt/data/jordiclive/scaled-rope


deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61500 /mnt/data/jordiclive/scaled-rope/finetune.py --output_dir saved_ckpts_32k --configs defaults lora-7b --deepspeed 2>&1 | tee debug_falcon.txt