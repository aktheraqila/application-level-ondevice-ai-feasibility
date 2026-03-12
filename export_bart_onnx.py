import os
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

MODEL_NAME = "facebook/bart-base"
ONNX_SAVE_DIR = "onnx_bart_fp32"

print(f"Starting ONNX export for {MODEL_NAME}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Export PyTorch model → ONNX
print("Exporting model to ONNX format...")
model = ORTModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    export=True,
    provider="CPUExecutionProvider"
)

# Save exported ONNX model
os.makedirs(ONNX_SAVE_DIR, exist_ok=True)
model.save_pretrained(ONNX_SAVE_DIR)
tokenizer.save_pretrained(ONNX_SAVE_DIR)

print("ONNX export completed successfully.")
print(f"Files saved in: {ONNX_SAVE_DIR}")