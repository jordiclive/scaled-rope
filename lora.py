from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def peft_model(
    model, peft_config, model_name, gradient_checkpointing=True
):

    if "falcon" in model_name:
        target_modules = ["dense_4h_to_h", "dense", "query_key_value", "dense_h_to_4h"]

    elif "llama" in model_name:
        target_modules = [
            "down_proj",
            "k_proj",
            "q_proj",
            "gate_proj",
            "o_proj",
            "up_proj",
            "v_proj",
        ]
    else:
        raise ValueError(
            f"Invalid model name '{model_name}'. The model name should contain 'falcon' or 'llama'"
        )
    config = LoraConfig(
        r=peft_config["r"],
        lora_alpha=peft_config["alpha"],
        target_modules=target_modules,
        lora_dropout=peft_config["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    if gradient_checkpointing:
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=gradient_checkpointing)
    model.print_trainable_parameters()
    return model


def load_peft_finetuned_model(model, peft_model_path):
    adapters_weights = torch.load(
        Path(peft_model_path).joinpath("adapter_model.bin"), map_location=model.device
    )
    model.load_state_dict(adapters_weights, strict=False)
    return model
