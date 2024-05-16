from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_model import load_model, kaiming_initialization

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        input_text: str = "Once upon a time, ",
        max_new_tokens: int = 16
):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    # print(outputs)
    print(generated_text)


def inference_my_model(
        input_text: str = "Once upon a time, ",
        max_new_tokens: int = 16,
        model_name: str = "saves/checkpoint-101000",
):
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    # print(outputs)
    print(generated_text)


if __name__ == '__main__':
    # model, tokenizer = load_model()
    # kaiming_initialization(model)
    # inference(model, tokenizer)
    inference_my_model(input_text="Once upon a time, in a beautiful garden, there lived a little rabbit named Peter Rabbit.", max_new_tokens=256)
