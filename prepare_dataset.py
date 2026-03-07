from datasets import load_dataset
import os

print("Loading CNN/DailyMail dataset...")

dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

os.makedirs("dataset", exist_ok=True)

short_count = 0
medium_count = 0
long_count = 0

for item in dataset:

    article = item["article"]
    words = article.split()

    if len(words) > 1200:

        # SHORT (150 words)
        if short_count < 40:
            text = " ".join(words[:150])
            with open(f"dataset/short_{short_count+1}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            short_count += 1

        # MEDIUM (600 words)
        if medium_count < 40:
            text = " ".join(words[:600])
            with open(f"dataset/medium_{medium_count+1}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            medium_count += 1

        # LONG (1200 words)
        if long_count < 40:
            text = " ".join(words[:1200])
            with open(f"dataset/long_{long_count+1}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            long_count += 1

    if short_count == 40 and medium_count == 40 and long_count == 40:
        break

print("Dataset created successfully.")
print("Short:", short_count)
print("Medium:", medium_count)
print("Long:", long_count)