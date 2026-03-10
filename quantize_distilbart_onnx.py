import os
import shutil
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_DIR = "onnx_distilbart_fp32"
SAVE_DIR = "onnx_distilbart_int8"

if os.path.isfile(SAVE_DIR):
    os.remove(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.save_pretrained(SAVE_DIR)

for item in os.listdir(MODEL_DIR):
    if item.endswith(".json"):
        shutil.copy2(os.path.join(MODEL_DIR, item), os.path.join(SAVE_DIR, item))

qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
onnx_files = ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]

for file_name in onnx_files:
    quantizer = ORTQuantizer.from_pretrained(MODEL_DIR, file_name=file_name)
    quantizer.quantize(save_dir=SAVE_DIR, quantization_config=qconfig)
    
    quantized_path = os.path.join(SAVE_DIR, file_name.replace(".onnx", "_quantized.onnx"))
    target_path = os.path.join(SAVE_DIR, file_name)
    
    if os.path.exists(quantized_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(quantized_path, target_path)