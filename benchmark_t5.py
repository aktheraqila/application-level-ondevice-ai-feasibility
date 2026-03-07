import os
import time
import psutil
import pandas as pd

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM


# -----------------------------
# CONFIGURATION
# -----------------------------

DATASET_FOLDER = "dataset"
RESULT_FILE = "benchmark_results.csv"

MODEL_PATH = "onnx_model"

WARMUP_RUNS = 5
MEASURED_RUNS = 10

MIN_LENGTH = 50
MAX_LENGTH = 50


# -----------------------------
# LOAD TOKENIZER
# -----------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")


# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading ONNX model...")
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


# -----------------------------
# LOAD DATASET
# -----------------------------

print("Loading dataset...")

articles = []

for file in sorted(os.listdir(DATASET_FOLDER)):

    if file.endswith(".txt"):

        path = os.path.join(DATASET_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Detect article category
        if file.startswith("short"):
            category = "short"
        elif file.startswith("medium"):
            category = "medium"
        else:
            category = "long"

        articles.append((file, text, category))

print("Total articles:", len(articles))


# -----------------------------
# PREPARE STORAGE
# -----------------------------

results = []

process = psutil.Process(os.getpid())


# -----------------------------
# BENCHMARK LOOP
# -----------------------------

for idx, (article_id, text, category) in enumerate(articles):

    print("Processing:", article_id)

    # -----------------------------
    # WARMUP RUNS
    # -----------------------------

    for _ in range(WARMUP_RUNS):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        _ = model.generate(
            **inputs,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH
        )

    # -----------------------------
    # MEASURED RUNS
    # -----------------------------

    for run in range(MEASURED_RUNS):

        # TOKENIZATION TIME
        token_start = time.time()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        tokenization_time = time.time() - token_start

        input_tokens = inputs["input_ids"].shape[1]

        # CPU / MEMORY BEFORE
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = process.memory_info().rss

        # MODEL INFERENCE
        start_time = time.time()

        outputs = model.generate(
            **inputs,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH
        )

        total_latency = time.time() - start_time

        # CPU / MEMORY AFTER
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = process.memory_info().rss

        memory_mb = max(mem_before, mem_after) / (1024 * 1024)

        output_tokens = outputs.shape[1]

        # DERIVED METRICS
        tokens_per_second = output_tokens / total_latency
        latency_per_token = total_latency / output_tokens

        results.append({
            "model": "t5-small",
            "runtime": "onnx_int8",
            "article_index": idx,
            "article_id": article_id,
            "article_length_category": category,
            "run_id": run,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokenization_time": tokenization_time,
            "total_latency": total_latency,
            "tokens_per_second": tokens_per_second,
            "latency_per_token": latency_per_token,
            "memory_mb": memory_mb,
            "cpu_percent": max(cpu_before, cpu_after)
        })


# -----------------------------
# SAVE RESULTS
# -----------------------------

df = pd.DataFrame(results)

df.to_csv(RESULT_FILE, index=False)

print("Benchmark finished.")
print("Results saved to:", RESULT_FILE)