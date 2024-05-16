from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

repo_name = 'TinyStories-LLaMA2-20M-256h-4l-GQA'
model_name = 'saves/checkpoint-101000'
access_token = ""
login(token=access_token, add_to_git_credential=True)  # 输入 Access Tokens
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
print('Pushed to Hugging Face Hub')
