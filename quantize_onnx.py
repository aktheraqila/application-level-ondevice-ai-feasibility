from onnxruntime.quantization import quantize_dynamic, QuantType

print("Quantizing encoder...")
quantize_dynamic(
    "onnx_model/encoder_model.onnx",
    "onnx_model/encoder_model_int8.onnx",
    weight_type=QuantType.QInt8
)

print("Quantizing decoder...")
quantize_dynamic(
    "onnx_model/decoder_with_past_model.onnx",
    "onnx_model/decoder_with_past_model_int8.onnx",
    weight_type=QuantType.QInt8
)

print("Quantization complete.")