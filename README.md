# Swin Transformer Edge Optimization

Systematic optimization of Swin Transformer achieving **4.1x inference speedup** (32ms → 7.73ms) on NVIDIA Jetson Orin through profiling-guided optimization, TensorRT conversion, and INT8 quantization.

!["Performance_Comparison"](./edge_optimization/results/swin_performance_comparison.png)
## Results Summary

| Configuration | Latency | Speedup | Throughput | Model Size | Top-1 Acc | Top-5 Acc |
|--------------|---------|---------|------------|------------|-----------|-----------|
| PyTorch FP32 | 32.0 ms | 1.0x | 31.3 FPS | 110 MB | 79.80% | 94.30% |
| TensorRT FP32 | 9.07 ms | 3.5x | 110.3 FPS | 14.9 MB | - | — |
| Naive TensorRT INT8 | 7.73 ms | **4.1x** | **129.4 FPS** | **28 MB** | 5.5% | 20% |
| SmoothQuant INT8 | 7.73 ms | **4.1x** | **129.4 FPS** | **28 MB** | **78.14%** | **94.54%** |

**Key Achievements:**
- 4.1x inference speedup enables real-time perception
- 4x model compression (110MB → 28MB)
- 129 FPS throughput (real-time capable for 60+ Hz systems)
- Memory bandwidth bottleneck identified and analyzed
- SmoothQuant PTQ achieves 78.14% top-1 / 94.54% top-5 accuracy with INT8 quantization

---

## Table of Contents

- [Motivation](#motivation)
- [Methodology](#methodology)
- [Profiling & Analysis](#profiling--analysis)
- [Optimization Pipeline](#optimization-pipeline)
- [Results](#results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Technologies](#technologies)

---

## Motivation

Vision-Language-Action (VLA) models are becoming critical for autonomous systems, requiring efficient vision encoding for real-time performance. This project systematically evaluates and optimizes the vision encoding component of VLA frameworks to identify performance bottlenecks and optimization opportunities for deployment on NVIDIA Jetson Orin.

**Why Swin Transformer?**
1. **Industry adoption:** Widely used in autonomous driving perception (BEVFormer, PETR, BEVDet)
2. **Transformer architecture:** Self-attention mechanisms represent core building blocks of modern VLA models
3. **Edge deployment relevance:** Hierarchical design suitable for resource-constrained devices

---

## Methodology

### 1. Baseline Measurement
**Setup:**
- Model: Swin-Tiny (smallest variant for edge deployment)
- Platform: NVIDIA Jetson Orin
- Input: 224×224×3 RGB images
- Framework: PyTorch with CUDA

**Baseline Results:**
```
PyTorch FP32 GPU: 32.0 ms per image (31.3 FPS)
```

**Initial Observation:** Too slow for real-time perception (need <15ms for 60Hz)

### 2. Profiling & Bottleneck Analysis

**Two-level profiling approach:**

#### GPU-Level Profiling (Nsight Systems)
- Captured end-to-end inference execution
- Identified GEMM (matrix multiplication) operations as primary GPU workload
- Confirmed compute-bound behavior (minimal idle time)

#### Layer-Wise Profiling (Custom CUDA Events)
- Implemented timing hooks for each layer type
- Measured time distribution across operations

**Key Finding:** **87% of inference time in Linear layers** (MLP + Attention projections)

| Layer Type | Time (ms) | Percentage | Quantizable? |
|------------|-----------|------------|--------------|
| MLP FC1 (Expand) | 23.19 | 27.5% | ✅ |
| MLP FC2 (Project) | 23.30 | 27.6% | ✅ |
| Attention QKV | 17.67 | 21.0% | ✅ |
| Attention Proj | 9.55 | 11.3% | ✅ |
| LayerNorm | 5.45 | 6.5% | ❌ |
| Patch Embed | 4.77 | 5.7% | ✅ |
| Other | 0.39 | 0.5% | - |

**Critical Insight:** Matrix multiplication operations in MLP and Attention are the bottleneck → quantization and graph optimization should be effective.

!["Layer_profile](./edge_optimization/results/layer_profile_bar.png)

---

## Optimization Pipeline

### Stage 1: PyTorch Quantization Exploration

**Approach:** Applied PyTorch dynamic quantization (INT8) to Linear layers

**Results:**
| Configuration | Device | Latency | Notes |
|--------------|--------|---------|-------|
| FP32 | GPU | 32.0 ms | Baseline |
| FP32 | CPU | 578 ms | Reference |
| INT8 | CPU | 442 ms | 1.31x vs CPU FP32 |

**Learning:** PyTorch quantization only works on CPU (not GPU-accelerated). Need TensorRT for GPU INT8.

**Decision:** Skip PyTorch quantization → proceed directly to TensorRT

### Stage 2: TensorRT Graph Optimization

**Approach:**
1. Export PyTorch model to ONNX (opset 17)
2. Convert ONNX to TensorRT engine with graph optimization
3. Benchmark on Jetson Orin

**TensorRT Optimizations Applied:**
- Kernel fusion (combine multiple ops)
- Memory layout optimization
- Constant folding
- Dead code elimination

**Results:**
```
PyTorch FP32:   32.0 ms (baseline)
TensorRT FP32:   9.07 ms (3.5x speedup)
```

**Impact:** Graph optimization alone provided 3.5x speedup!

### Stage 3: INT8 Quantization

**Approach:**
- Applied calibration-based INT8 quantization
- TensorRT automatically determines optimal quantization scales
- GPU-native INT8 Tensor Core acceleration

**Results:**
```
TensorRT FP32:  9.07 ms
TensorRT INT8:  7.73 ms (1.17x additional speedup)
```

**Total Speedup:** 32.0ms → 7.73ms = **4.1x overall**

### Stage 4: SmoothQuant Post-Training Quantization

**Motivation:** Standard INT8 quantization suffers accuracy loss due to large activation outliers in transformer attention layers. SmoothQuant migrates quantization difficulty from activations to weights via a mathematically equivalent per-channel scaling transform.

**Approach:**
- Applied SmoothQuant to all Linear layers (MLP + Attention projections)
- Per-channel migration scale computed from calibration data
- Weights absorb the scaling factor offline; activations become easier to quantize
- Evaluated on ImageNet validation set (50K images)

**Results:**
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **78.14%** |
| Top-5 Accuracy | **94.54%** |
| Precision | INT8 |
| Latency | 7.73 ms (unchanged) |

**Analysis:** SmoothQuant preserves the full 4.1x speedup from TensorRT INT8 while providing a measurable accuracy baseline. The accuracy drop relative to FP32 is a direct consequence of PTQ without fine-tuning — see Future Work for QAT as the next step.

### Stage 5: Batch Processing Analysis

**Motivation:** Real-world systems may process multiple images (multi-camera feeds, batched inference)

**Results:**

| Batch Size | FP32 Total | INT8 Total | Per-Image (INT8) | Throughput | INT8 Speedup |
|-----------|------------|------------|------------------|------------|--------------|
| 1 | 9.07 ms | 7.73 ms | 7.73 ms | 129.4 FPS | 1.17x |
| 4 | 25.72 ms | 23.82 ms | 5.96 ms | 167.9 FPS | 1.08x |
| 8 | 48.37 ms | 46.77 ms | 5.85 ms | 171.1 FPS | 1.03x |

**Observations:**
- ✅ Batch processing improves per-image throughput by 24%
- ⚠️ INT8 speedup **decreases** with larger batches (1.17x → 1.03x)

**Root Cause:** Memory bandwidth bottleneck
- Larger batches = more data movement
- Memory bandwidth becomes limiting factor
- INT8 compute advantage diminished by memory constraints

---

## Results

### Performance Summary

<table>
<tr>
<td>

**Latency Reduction**
- Baseline: 32.0 ms
- Optimized: 7.73 ms
- **Improvement: 4.1x**

</td>
<td>

**Throughput Increase**
- Baseline: 31.3 FPS
- Optimized: 129.4 FPS
- **Improvement: 4.1x**

</td>
</tr>
<tr>
<td>

**Model Compression**
- FP32 PyTorch: 110 MB
- INT8 PyTorch: 28 MB
- **Compression: 4x**

</td>
<td>

**TensorRT Engine**
- FP32 Engine: 14.9 MB
- INT8 Engine: 13.2 MB
- **Compression: 8.3x**

</td>
</tr>
</table>

### Detailed Performance Breakdown

See [edge-optimization/results/](edge-optimization/results/) for:
- Complete performance comparison charts
- Nsight Systems profiling visualizations
- Layer-wise timing breakdowns
- Batch processing analysis
- Memory bandwidth analysis

---

## Key Findings

### 1. Graph Optimization > Quantization on Edge Devices

**TensorRT graph optimization: 3.5x speedup**
- Kernel fusion and memory layout optimization
- Significantly more impactful than quantization alone

**INT8 quantization: 1.17x additional speedup**
- Still valuable but secondary to graph optimization
- Limited by memory bandwidth on edge hardware

**Lesson:** Tool selection and graph-level optimization matter more than low-level manual tuning

### 2. Memory Bandwidth is the Bottleneck

**Evidence:**
- INT8 speedup decreases with larger batches (1.17x → 1.03x)
- Per-image latency improves with batching (7.73ms → 5.85ms)
- But memory transfer dominates compute time

**Implication:**
- Edge devices are memory-bound, not compute-bound for transformers
- Architecture decisions should minimize data movement
- Batch processing helps throughput but doesn't scale linearly

### 3. Systematic Profiling Enables Data-Driven Optimization

**Profiling identified:**
- Specific bottlenecks (87% time in Linear layers)
- Optimization opportunities (GEMM operations)
- Hardware constraints (memory bandwidth)

**Result:** Focused optimization efforts on high-impact areas, avoided premature optimization

---

## Repository Structure
```
Swin-Transformer/
├── README.md                    # This file (optimization overview)
├── edge-optimization/           # My optimization work
│   ├── scripts/                # Profiling & optimization scripts
│   │   ├── profile_baseline.py
│   │   ├── export_onnx.py
│   │   ├── benchmark_tensorrt.sh
│   │   └── visualize_results.py
│   ├── results/                # Performance data & visualizations
│   │   ├── results.csv
│   │   ├── comprehensive_results.png
│   │   └── nsight_profiles/
│   └── models/                 # Optimized model artifacts
│       ├── swin_tiny_fp32.onnx
│       ├── swin_fp32.trt
│       └── swin_int8.trt
└── original/                   # Original Swin Transformer implementation
    └── [Microsoft's original code]
```

---

## Quick Start

### Prerequisites
- NVIDIA Jetson Orin with JetPack 6.2
- PyTorch 2.0+
- TensorRT 8.6+
- ONNX Runtime

### Running the Optimization Pipeline
```bash
# 1. Profile baseline PyTorch model
cd edge-optimization/scripts
python profile_baseline.py

# 2. Export to ONNX
python export_onnx.py

# 3. Convert to TensorRT and benchmark
./benchmark_tensorrt.sh

# 4. Generate performance visualizations
python visualize_results.py
```

### Expected Output
```
PyTorch FP32:   32.0 ms
TensorRT FP32:   9.07 ms (3.5x speedup)
TensorRT INT8:   7.73 ms (4.1x speedup)
```

---

## Technologies

**Profiling:**
- NVIDIA Nsight Systems (GPU profiling)
- CUDA Events (layer-wise timing)
- PyTorch Profiler

**Optimization:**
- TensorRT (graph optimization + quantization)
- ONNX (model export format)
- INT8 post-training quantization
- SmoothQuant (activation-weight migration for accurate INT8)

**Deployment:**
- NVIDIA Jetson Orin
- JetPack 6.2
- CUDA 12.2

**Analysis:**
- Python (NumPy, Pandas)
- Matplotlib (visualizations)

---

## Applications

This optimization methodology applies to:

1. **Autonomous Driving Perception**
   - BEVFormer (3D object detection)
   - PETR (transformer-based detection)
   - Multi-camera perception systems

2. **Vision-Language Models**
   - Real-time VLM inference on edge
   - Efficient vision encoding for VLA systems
   - Multi-modal transformer architectures

3. **Edge AI Deployment**
   - Resource-constrained devices
   - Real-time inference requirements
   - Low-power operation

---

## Future Work

- [x] **SmoothQuant PTQ** — 78.14% top-1 / 94.54% top-5 with INT8 (✅ done)
- [ ] **Quantization-Aware Training (QAT)** for further accuracy recovery beyond SmoothQuant PTQ
- [ ] **Apply to BEVFormer** for 3D perception optimization
- [ ] **Mixed precision strategies** (FP16 attention, INT8 MLP)
- [ ] **Structured pruning** for additional speedup
- [ ] **Knowledge distillation** (large → small model)
- [ ] **Multi-model deployment** strategies for full VLA pipeline

---

## References

### Original Work
This project builds upon Microsoft's Swin Transformer:
- Paper: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- Code: [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

### Citation
```bibtex
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

---

## License

This optimization work builds upon the original Swin Transformer implementation, which is licensed under the Apache License 2.0.

---

## Author

**Your Name**
- Email: goyalaarju@gmail.com
- LinkedIn: [Aarju Goyal](https://www.linkedin.com/in/aarju/)
- GitHub: [@AarjuGoyal](https://github.com/AarjuGoyal)

*Developed as part of research into efficient transformer inference for autonomous driving perception systems.*

---

## Acknowledgments

- Microsoft Research for the original Swin Transformer implementation
- NVIDIA for TensorRT and Jetson platform
- Volkswagen Group of America for supporting this research

EOF