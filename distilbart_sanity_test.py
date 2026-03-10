from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

model_name = "sshleifer/distilbart-cnn-6-6"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()  # Consistent with your benchmark_t5_pytorch_fp32.py

text = """
The Apollo program was the third United States human spaceflight program carried out by NASA,
which succeeded in landing the first humans on the Moon from 1969 to 1972.
"""

print("Tokenizing input...")
inputs = tokenizer(text, return_tensors="pt", truncation=True)

print("Generating summary...")
start_time = time.time() # Consistent with your benchmark timing

with torch.no_grad(): # Consistent with your benchmark memory safety
    summary_ids = model.generate(
        **inputs, 
        min_new_tokens=20, 
        max_new_tokens=50,
        do_sample=False
    )

latency = time.time() - start_time
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"\nLatency: {latency:.2f}s")
print("Summary:")
print(summary)