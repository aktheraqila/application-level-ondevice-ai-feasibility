from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

print("Loading ONNX model...")

model = ORTModelForSeq2SeqLM.from_pretrained("onnx_model")

text = """
summarize: Artificial intelligence is transforming many industries by enabling machines to perform tasks that normally require human intelligence.
"""

print("Tokenizing input...")
inputs = tokenizer(text, return_tensors="pt")

print("Generating summary...")
outputs = model.generate(**inputs, max_length=40)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated summary:")
print(summary)
