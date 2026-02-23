# Application-Level Feasibility of On-Device AI for Mobile Applications

## Research Objective

This project investigates whether transformer-based language models can be practically deployed inside real Android applications running entirely on-device, without internet connectivity.

The study evaluates feasibility at the application level, not just at the model level.

---

## Research Gap

Prior work primarily focuses on:
- Model compression techniques
- Architecture optimization
- Hardware acceleration

However, limited work evaluates:
- Real Android app integration
- End-to-end latency in production-like scenarios
- Memory, CPU, battery, and thermal impact on actual smartphones

This study addresses that gap.

---

## Experimental Design

### Phase 1 — Model Preparation
- Export transformer model to ONNX
- Apply dynamic INT8 quantization
- Reduce model size for mobile feasibility

### Phase 2 — Android Integration
- Integrate model into a simple Android app
- Ensure fully offline inference

### Phase 3 — Real Device Evaluation
Tested on:
- Honor 9X (Kirin 710F)
- Realme C17 (Snapdragon 460)
- Tecno Camon 40 Pro 4G (Helio G100)

Measured metrics:
- Inference latency
- Memory usage
- CPU utilization
- Battery consumption
- Device temperature

---

## Model Optimization Results

Dynamic INT8 quantization achieved:

| Component | FP32 | INT8 |
|-----------|------|------|
| Encoder   | 138 MB | 34 MB |
| Decoder   | 214 MB | 54 MB |
| Total     | 352 MB | 88 MB |

~75% reduction in total model size.

Quantization enabled feasible mobile deployment.

---

## Expected Outcome

This research aims to determine:
- Acceptable performance thresholds for on-device AI
- Practical hardware limitations
- Feasibility differences across low-end and mid-range devices

---

## Repository Contents

- ONNX models (INT8)
- Quantization scripts
- Android integration (planned)
- Benchmarking scripts (planned)

---

## Status

✔ Model export complete  
✔ Quantization complete  
⬜ Android benchmarking in progress  
⬜ Full experimental results pending  
