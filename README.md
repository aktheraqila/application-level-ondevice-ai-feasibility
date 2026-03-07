# Application-Level On-Device AI Feasibility Study

## Overview

This repository contains the experimental framework for the research study:

**"Application-Level Feasibility of On-Device AI for Mobile Applications"**

The objective of this research is to evaluate whether transformer-based language models can be practically deployed inside real Android applications running fully on-device, without internet connectivity.

Unlike prior work that focuses primarily on model compression or architectural optimization, this study evaluates feasibility at the *application level*, considering real-world system constraints on actual smartphones.

---

## Research Motivation

Recent advances in lightweight transformer models enable on-device inference. However, limited research evaluates:

- End-to-end Android app integration
- Real-device latency under CPU-only execution
- Memory footprint in production-like scenarios
- Battery consumption impact
- Thermal behavior under repeated inference

This project addresses that gap through real-device experimentation.

---

## Methodology

The research is conducted in three major phases:

### Phase 1 – Model Preparation
- Export pretrained transformer model to ONNX format
- Apply dynamic INT8 quantization using ONNX Runtime
- Reduce model size for mobile deployment feasibility

### Phase 2 – Android Integration
- Integrate quantized model into a simple Android application
- Ensure fully offline execution (no internet usage)
- CPU-based inference

### Phase 3 – Experimental Evaluation
Experiments are conducted on representative low-end and mid-range smartphones:

- Honor 9X (Kirin 710F)
- Realme C17 (Snapdragon 460)
- Tecno Camon 40 Pro 4G (Helio G100)

Measured metrics:
- Inference latency
- Memory usage
- CPU utilization
- Battery consumption
- Device temperature variation

---

## Model Optimization Results

Dynamic INT8 quantization achieved approximately 75% model size reduction:

| Component | FP32 | INT8 |
|-----------|------|------|
| Encoder   | 138 MB | 34 MB |
| Decoder   | 214 MB | 54 MB |
| **Total** | **352 MB** | **88 MB** |

Quantization significantly improves deployability while maintaining functional correctness.

---

## Research Contribution

This study contributes:

- Application-level empirical evaluation of on-device AI
- Real-device experimentation on low-end hardware
- System-level feasibility analysis (latency, memory, battery, thermal)
- Identification of practical deployment thresholds

---

## Repository Structure
