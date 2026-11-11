from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './local/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
# 打印模型的参数数量 124439808-->124M
print(sum(p.numel() for p in model.parameters()))
