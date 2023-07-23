from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def find_all_linear_names(bits, model):
    if bits is not None:
        import bitsandbytes as bnb
        if bits == 4:
            cls = bnb.nn.Linear4bitLt
        elif bits == 8:
            cls = bnb.nn.Linear8bitLt
    else:
        cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def peft_model(
    model, peft_config, gradient_checkpointing=True, bits=None
):
    # linear_names = find_all_linear_names(bits, model)
    linear_names = ['gate_proj', 'q_proj', 'k_proj', 'v_proj', 'down_proj', 'o_proj', 'up_proj']
    print('linear_names', linear_names)
    print('PEFT config', peft_config)
    config = LoraConfig(
        r=peft_config["r"],
        lora_alpha=peft_config["alpha"],
        target_modules=linear_names,
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
