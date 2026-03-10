from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
import os

MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
SAVE_DIR = "onnx_distilbart_fp32"

print(f"Starting professional ONNX export for {MODEL_NAME}...")

# This single line handles the Encoder, Decoder, and the KV-Cache wiring
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAME, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Save the optimized structure to your project folder
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nSuccess! DistilBART ONNX model saved in: {SAVE_DIR}")
print("Check your folder; you should see encoder_model.onnx and decoder_model.onnx.")