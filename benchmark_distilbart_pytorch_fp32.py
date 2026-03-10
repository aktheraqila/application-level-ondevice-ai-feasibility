import os
import time
import psutil
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATASET_FOLDER = "dataset"
RESULT_FILE = "benchmark_distilbart_pytorch_fp32.csv"
MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
WARMUP_RUNS = 5
MEASURED_RUNS = 10
MIN_LENGTH = 20
MAX_LENGTH = 50

print(f"Loading PyTorch FP32 model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

articles = []
for file in sorted(os.listdir(DATASET_FOLDER)):
    if file.endswith(".txt"):
        path = os.path.join(DATASET_FOLDER, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        category = file.split('_')[0] 
        articles.append((file, text, category))

results = []
process = psutil.Process(os.getpid())

print(f"Starting PyTorch FP32 benchmark on {len(articles)} articles...")

with torch.no_grad():
    for idx, (article_id, text, category) in enumerate(articles):
        print(f"[{idx+1}/{len(articles)}] Benchmarking {article_id}...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_tokens = inputs["input_ids"].shape[1]

        for _ in range(WARMUP_RUNS):
            _ = model.generate(**inputs, min_new_tokens=MIN_LENGTH, max_new_tokens=MAX_LENGTH)

        for run in range(MEASURED_RUNS):
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = process.memory_info().rss
            
            start_time = time.time()
            outputs = model.generate(**inputs, min_new_tokens=MIN_LENGTH, max_new_tokens=MAX_LENGTH)
            total_latency = time.time() - start_time
            
            cpu_after = psutil.cpu_percent(interval=None)
            mem_after = process.memory_info().rss

            output_tokens = outputs.shape[1]
            results.append({
                "model": "distilbart-6-6",
                "runtime": "pytorch_fp32",
                "article_id": article_id,
                "category": category,
                "run_id": run,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_latency": total_latency,
                "tokens_per_second": output_tokens / total_latency,
                "memory_mb": max(mem_before, mem_after) / (1024 * 1024),
                "cpu_percent": max(cpu_before, cpu_after)
            })

pd.DataFrame(results).to_csv(RESULT_FILE, index=False)
print(f"Done! Results saved to {RESULT_FILE}")