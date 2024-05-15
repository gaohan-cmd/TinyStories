import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# 初始化模型
def load_model():
    os.environ['WANDB_DISABLED'] = 'true'  # 禁用 wandb，也可以不用这一条
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    hidden_size = 256
    intermediate_size = (int(hidden_size * 8 / 3 / 128) + 1) * 128

    config = AutoConfig.for_model(
        model_type="llama",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=16,
        num_hidden_layers=4,
        num_key_value_heads=8
    )
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float32
    ).to(device)
    return model, tokenizer


# Kaiming 初始化
def kaiming_initialization(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in name:
            # 一般偏置项可以初始化为0
            torch.nn.init.constant_(param, 0)


# 打印模型的每一层及其参数大小
def print_model_parameters(model):
    print("Layer Name & Parameters")
    print("----------------------------")
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        total_params += param_count
        print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
    print("----------------------------")
    print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")

    print_model_parameters(model)
