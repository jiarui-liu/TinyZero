from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = (
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # or whichever Qwen2 model you're using
    "Qwen/Qwen2.5-1.5B"
)
# save_path = "/compute/babel-4-33/jiaruil5/.cache/DeepSeek-R1-Distill-Qwen-1.5B"
save_path = "/compute/babel-4-33/jiaruil5/.cache/Qwen/Qwen2.5-1.5B"

# Download and save the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
