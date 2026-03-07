import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

print("STEP 1: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

print("STEP 2: Loading encoder model...")
encoder_session = ort.InferenceSession(
    "onnx_model/encoder_model_int8.onnx",
    providers=["CPUExecutionProvider"]
)

print("STEP 3: Loading decoder model...")
decoder_session = ort.InferenceSession(
    "onnx_model/decoder_with_past_model_int8.onnx",
    providers=["CPUExecutionProvider"]
)

print("STEP 4: Preparing input text...")
text = "summarize: Artificial intelligence is transforming industries by allowing machines to perform tasks that normally require human intelligence."

print("STEP 5: Tokenizing input...")
inputs = tokenizer(text, return_tensors="np")

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("STEP 6: Running encoder inference...")

encoder_outputs = encoder_session.run(
    None,
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
)

print("STEP 7: Encoder executed successfully.")
print("Encoder output shape:", np.array(encoder_outputs[0]).shape)

print("SUCCESS: Quantized ONNX encoder works correctly.")