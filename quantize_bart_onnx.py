import os
import shutil
from onnxruntime.quantization import quantize_dynamic, QuantType

ONNX_DIR = "onnx_bart_fp32"
QUANT_DIR = "onnx_bart_int8"

print("Starting dynamic INT8 quantization for BART-base")

os.makedirs(QUANT_DIR, exist_ok=True)

# Copy tokenizer/config files
for file in os.listdir(ONNX_DIR):

    src = os.path.join(ONNX_DIR, file)
    dst = os.path.join(QUANT_DIR, file)

    if not file.endswith(".onnx"):

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)

# Models to quantize
onnx_models = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx"
]

for model_file in onnx_models:

    input_path = os.path.join(ONNX_DIR, model_file)
    output_path = os.path.join(QUANT_DIR, model_file)

    if os.path.exists(input_path):

        print(f"Quantizing {model_file}...")

        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8
        )

print("\nINT8 quantization completed")
print(f"Quantized models saved in: {QUANT_DIR}")