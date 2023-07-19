# LoRA + Flash + DS 
This project adds LoRA, Flash attn patch and DS (Deepspeed) to [Scaled Rope](https://github.com/jquesnelle/scaled-rope).

## Setup and Installation

1. To install the necessary dependencies, use pip to install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

2. Update the `config.yaml` file as per your requirements.

## Usage

To run the application, use the following command:

```
python --configs defaults <your_override_config>
```

Replace `<your_override_config>` with your specific configuration specified in `configs/config.yaml`. Command line arguments can also be overridden.

**Please Note:** The current setup assumes linear interpolation and a Llama-based model, by setting `max_position_embeddings: 32768`. For example, if `interpolation_scale` is not set, it will be calculated as `2048 / config.max_position_embeddings = 0.0625`.

## Data
- Specify packed untokenized datasets on the hub under dataset_names e.g. (`Multi-Domain-Expert-Layers/the_pile_books3_packed_128k`)
- If pretokenized=True, specify a single pre-tokenized dataset on the hub under dataset_names (`conceptofmind/rp-packed-32k-no-filter` for OpenLLaMA)

### Running on a Single Node

Use the following commands for running on a single node:

1. Export the necessary paths:

    ```
    export PYTHONPATH="/mnt/data/jordiclive/scaled-rope:$PYTHONPATH"
    export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
    export HF_DATASETS_CACHE="/mnt/data/jordiclive/transformers_cache"
    export HF_HOME="/mnt/data/jordiclive/transformers_cache"
    export WANDB_API_KEY=""
    ```

2. Run the script:

    ```
    deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61500 finetune.py --output_dir saved_ckpts_32k --configs defaults lora-7b --deepspeed
    ```

### Running on Multiple Nodes

Example script: `scripts/juwels_booster.sh`
