# RESEARCH AND DEVELOPMENT OF A LIGHTWEIGHT REAL-TIME PLANT DISEASE DETECTION MODEL USING CUSTOM YOLOV8N WITH DEPTHWISE SEPARABLE AND LOW-RANK CONVOLUTIONS

## ACKNOWLEDGMENTS

The completion of this graduation project is the result of continuous study, implementation, and experimental evaluation in the field of real-time object detection for agricultural monitoring. The author would like to express sincere appreciation to the academic advisor, faculty members, and colleagues who provided technical guidance and valuable feedback during the development of the YOLOv8-based detection model.

Special thanks are also given to the open-source deep learning community and the Ultralytics YOLO framework, which provide a flexible foundation for implementing, modifying, training, and evaluating modern object detection models.

## TABLE OF CONTENTS

[**ACKNOWLEDGMENTS 1**](#acknowledgments)

[**ABSTRACT 7**](#abstract)

[**CHAPTER 1. INTRODUCTION 8**](#chapter-1-introduction)

[**1.1. Reasons for Choosing the Topic 8**](#11-reasons-for-choosing-the-topic)

[**1.2. Goal and Research Subjects 9**](#12-goal-and-research-subjects)

[**1.3. Expected Results 10**](#13-expected-results)

[**CHAPTER 2. THEORETICAL BACKGROUND 12**](#chapter-2-theoretical-background)

[**2.1. Theoretical Background of YOLO Object Detection 12**](#21-theoretical-background-of-yolo-object-detection)

[**2.2. Dataset 13**](#22-dataset)

[**2.3. YOLOv8n Model Overview 15**](#23-yolov8n-model-overview)

[**2.4. Key Techniques Used in Model Optimization 17**](#24-key-techniques-used-in-model-optimization)

[**2.4.1. Convolutional Neural Networks 17**](#241-convolutional-neural-networks)

[**2.4.2. Depthwise Separable Convolution 18**](#242-depthwise-separable-convolution)

[**2.4.3. Low-Rank Decomposition 19**](#243-low-rank-decomposition)

[**2.4.4. Tucker-Style Spatial Decomposition 20**](#244-tucker-style-spatial-decomposition)

[**2.4.5. Post-Training Quantization and Vitis-AI Deployment 21**](#245-post-training-quantization-and-vitis-ai-deployment)

[**CHAPTER 3. ENHANCING YOLOV8N MODEL 22**](#chapter-3-enhancing-yolov8n-model)

[**3.1. Custom YOLOv8n-DSC Model Overview 22**](#31-custom-yolov8n-dsc-model-overview)

[**3.2. Model Development Methods 23**](#32-model-development-methods)

[**3.2.1. DSC_LR_Conv 23**](#321-dsc_lr_conv)

[**3.2.2. LRFuseC2f 24**](#322-lrfusec2f)

[**3.2.3. LRTuckerC2f 25**](#323-lrtuckerc2f)

[**3.2.4. DSCDetect 26**](#324-dscdetect)

[**3.2.5. Architecture Comparison 27**](#325-architecture-comparison)

[**CHAPTER 4. IMPLEMENTATION ALGORITHM AND EXPERIMENTAL RESULTS 30**](#chapter-4-implementation-algorithm-and-experimental-results)

[**4.1. Training Configuration 30**](#41-training-configuration)

[**4.2. Baseline and Custom Model Results 31**](#42-baseline-and-custom-model-results)

[**4.3. Comparison and Analysis 33**](#43-comparison-and-analysis)

[**CHAPTER 5. REAL-TIME DEPLOYMENT DISCUSSION 39**](#chapter-5-real-time-deployment-discussion)

[**5.1. Edge Deployment Requirements 39**](#51-edge-deployment-requirements)

[**5.2. Inference Speed and Model Size 40**](#52-inference-speed-and-model-size)

[**5.3. Practical Deployment Considerations 41**](#53-practical-deployment-considerations)

[**CHAPTER 6. HIGH-PERFORMANCE DEPLOYMENT DIRECTION 43**](#chapter-6-high-performance-deployment-direction)

[**6.1. Hardware Acceleration Motivation 43**](#61-hardware-acceleration-motivation)

[**6.2. Quantization and Compilation Direction 44**](#62-quantization-and-compilation-direction)

[**6.3. Future Hardware Evaluation 45**](#63-future-hardware-evaluation)

[**CHAPTER 7. CONCLUSION 47**](#chapter-7-conclusion)

[**REFERENCES 48**](#references)

## LIST OF FIGURES

[***Figure 1. YOLOv8 model family and real-time object detection workflow 12***](#figure-1-yolov8-model-family-and-real-time-object-detection-workflow)

[***Figure 2. Overall YOLOv8n baseline architecture with P3/P4/P5 detection heads 15***](#figure-2-overall-yolov8n-baseline-architecture-with-p3p4p5-detection-heads)

[***Figure 3. Standard convolution and depthwise separable convolution comparison 18***](#figure-3-standard-convolution-and-depthwise-separable-convolution-comparison)

[***Figure 4. Low-rank 1x1 convolution decomposition: Cin -> rank -> Cout 19***](#figure-4-low-rank-1x1-convolution-decomposition-cin---rank---cout)

[***Figure 5. Tucker-style 3x3 convolution decomposition 20***](#figure-5-tucker-style-3x3-convolution-decomposition)

[***Figure 6. Post-training quantization workflow for ZCU102 deployment 21***](#figure-6-post-training-quantization-workflow-for-zcu102-deployment)

[***Figure 7. Custom YOLOv8n-DSC architecture overview 22***](#figure-7-custom-yolov8n-dsc-architecture-overview)

[***Figure 8. DSC_LR_Conv block structure 23***](#figure-8-dsc_lr_conv-block-structure)

[***Figure 9. LRFuseC2f block structure 24***](#figure-9-lrfusec2f-block-structure)

[***Figure 10. LRTuckerC2f block structure 25***](#figure-10-lrtuckerc2f-block-structure)

[***Figure 11. DSCDetect lightweight detection head 26***](#figure-11-dscdetect-lightweight-detection-head)

[***Figure 12. Training and evaluation workflow for YOLOv8n baseline and custom model 30***](#figure-12-training-and-evaluation-workflow-for-yolov8n-baseline-and-custom-model)

[***Figure 13. Comparison of parameter count and GFLOPs 33***](#figure-13-comparison-of-parameter-count-and-gflops)

[***Figure 14. Comparison of mAP50 and mAP50-95 34***](#figure-14-comparison-of-map50-and-map50-95)

[***Figure 15. Comparison of inference time and estimated FPS 35***](#figure-15-comparison-of-inference-time-and-estimated-fps)

[***Figure 16. Edge deployment workflow using ONNX, OpenVINO, TensorRT, or Vitis-AI 41***](#figure-16-edge-deployment-workflow-using-onnx-openvino-tensorrt-or-vitis-ai)

[***Figure 17. Future high-performance deployment direction on hardware accelerators 43***](#figure-17-future-high-performance-deployment-direction-on-hardware-accelerators)

## LIST OF TABLES

[***Table 1. Dataset and evaluation configuration 13***](#table-1-dataset-and-evaluation-configuration)

[***Table 2. Key modules in YOLOv8n baseline architecture 16***](#table-2-key-modules-in-yolov8n-baseline-architecture)

[***Table 3. Quantization and deployment formats for ZCU102-oriented workflow 21***](#table-3-quantization-and-deployment-formats-for-zcu102-oriented-workflow)

[***Table 4. Backbone architecture comparison between YOLOv8n baseline and custom YOLOv8n 27***](#table-4-backbone-architecture-comparison-between-yolov8n-baseline-and-custom-yolov8n)

[***Table 5. Head architecture comparison between YOLOv8n baseline and custom YOLOv8n 28***](#table-5-head-architecture-comparison-between-yolov8n-baseline-and-custom-yolov8n)

[***Table 6. Main experimental results of YOLOv8n baseline and custom YOLOv8n 31***](#table-6-main-experimental-results-of-yolov8n-baseline-and-custom-yolov8n)

[***Table 7. Compression and efficiency comparison 33***](#table-7-compression-and-efficiency-comparison)

[***Table 8. Detection accuracy comparison 34***](#table-8-detection-accuracy-comparison)

[***Table 9. Detection error and IoU comparison 36***](#table-9-detection-error-and-iou-comparison)

[***Table 10. Edge deployment optimization options 41***](#table-10-edge-deployment-optimization-options)

[***Table 11. Future hardware evaluation metrics 45***](#table-11-future-hardware-evaluation-metrics)

# ABSTRACT

This report presents the research and development of a lightweight real-time plant disease detection model based on a custom YOLOv8n architecture. The objective is to improve the deployment suitability of YOLOv8n for resource-constrained edge devices while maintaining competitive detection accuracy for plant disease monitoring. Unlike the previous YOLOv5-based report structure, this work focuses on YOLOv8n and introduces structured compression into high-cost network regions.

The proposed model, named `yolov8n_custom_dsc_p45`, preserves the early feature extraction layers of YOLOv8n and selectively replaces high-channel P4/P5 downsampling layers, selected C2f blocks, and the detection head with lightweight modules. The main modules include `DSC_LR_Conv`, `LRFuseC2f`, `LRTuckerC2f`, and `DSCDetect`. These modules combine depthwise separable convolution, low-rank channel decomposition, and Tucker-style spatial decomposition to reduce parameter count and computational cost.

Experimental results show that the custom YOLOv8n model reduces parameters from 3,005,843 to 676,811 and GFLOPs from 8.1 to 3.2 compared with the YOLOv8n baseline. The model weight size decreases from 6.3 MB to 1.7 MB, while estimated FPS increases from 196 to 204. The custom model achieves 0.930 mAP50 compared with 0.937 for the baseline, preserving 99.25% of the baseline mAP50. These results demonstrate that the proposed YOLOv8n custom architecture achieves an effective trade-off between detection accuracy and computational efficiency.

# CHAPTER 1. INTRODUCTION

## 1.1. Reasons for Choosing the Topic

Plant disease detection is an important task in modern agriculture because early identification of unhealthy regions can support timely treatment and reduce crop loss. Manual inspection is often time-consuming and difficult to scale, especially when monitoring large agricultural or forest areas. Computer vision and deep learning provide an effective solution by automatically detecting disease regions from images.

YOLO-based object detection models are suitable for this task because they perform localization and classification in a single forward pass. This gives YOLO models strong real-time potential. However, real-world deployment often requires models to operate on embedded or edge devices where memory, computation, and power are limited. Therefore, a model must not only achieve high accuracy but also remain compact and fast.

YOLOv8n is a lightweight object detection model, but the standard architecture still contains convolutional layers and detection heads that contribute to computational cost. This motivates the development of a custom YOLOv8n model that reduces parameters and GFLOPs while preserving the accuracy required for plant disease detection.

## 1.2. Goal and Research Subjects

The main goal of this project is to design, implement, and evaluate a lightweight YOLOv8n-based model for real-time plant disease detection. The research focuses on a two-stage optimization pipeline: first, reducing the model complexity through structured architectural compression; second, preparing the compressed model for edge deployment through quantization and compilation for hardware platforms such as ZCU102.

The main research subjects include:

1. The YOLOv8n object detection architecture.
2. Depthwise separable convolution for reducing spatial convolution cost.
3. Low-rank decomposition for reducing channel mixing cost.
4. Tucker-style decomposition for lightweight 3x3 convolution approximation.
5. A lightweight detection head based on `DSCDetect`.
6. Post-training quantization and deployment preparation for ZCU102-oriented acceleration.
7. Experimental comparison between YOLOv8n baseline and the custom YOLOv8n model.

## 1.3. Expected Results

The expected outcome is a custom YOLOv8n model that maintains competitive detection accuracy while significantly reducing model complexity. The model is expected to:

- reduce parameter count and model weight size;
- reduce GFLOPs for faster inference;
- maintain mAP50 close to the YOLOv8n baseline;
- improve suitability for real-time deployment on edge devices;
- provide a clear architecture that can be further combined with INT8 quantization and ZCU102/Vitis-AI hardware acceleration.

# CHAPTER 2. THEORETICAL BACKGROUND

## 2.1. Theoretical Background of YOLO Object Detection

YOLO, short for You Only Look Once, is a family of single-stage object detection models. Unlike two-stage detectors that first generate region proposals and then classify them, YOLO performs bounding box regression and classification in one network pass. This design allows YOLO models to achieve high inference speed while maintaining strong detection accuracy.

YOLOv8 improves the flexibility and performance of earlier YOLO models through a modular architecture. The model includes a convolutional backbone for feature extraction, a neck for multi-scale feature fusion, and a detection head for bounding box prediction and classification. YOLOv8n is the nano version of YOLOv8, designed for lightweight and fast inference.

In this project, YOLOv8n is selected as the baseline because it provides a strong balance between speed and accuracy. The custom model keeps the main YOLOv8n topology but modifies selected high-cost layers to improve deployment efficiency.

## 2.2. Dataset

The plant disease detection task uses a single target class, represented by `nc: 1` in the model configuration. The evaluation contains 12,655 instances according to the experimental result file. The dataset is used to compare the standard YOLOv8n baseline with the proposed custom YOLOv8n model under the same training setting.

The detection problem is challenging because plant disease regions can appear in different sizes and densities. A suitable model must preserve enough low-level and multi-scale information to detect small or visually ambiguous targets.

## 2.3. YOLOv8n Model Overview

The baseline YOLOv8n architecture contains three main parts:

1. Backbone: extracts hierarchical visual features from the input image.
2. Neck: fuses multi-scale features using upsampling, concatenation, and C2f blocks.
3. Detection head: predicts bounding boxes and class probabilities at P3, P4, and P5 feature scales.

The standard YOLOv8n configuration uses Conv, C2f, SPPF, Concat, Upsample, and Detect modules. The P3 output is important for small objects, while P4 and P5 capture higher-level semantic features.

## 2.4. Key Techniques Used in Model Optimization

### 2.4.1. Convolutional Neural Networks

Convolutional Neural Networks use learnable kernels to extract spatial features from images. Standard convolutions can be computationally expensive when both input and output channel numbers are high. In object detection models, repeated convolutional operations account for a large portion of parameters and GFLOPs.

### 2.4.2. Depthwise Separable Convolution

Depthwise separable convolution decomposes a standard convolution into a depthwise convolution and a pointwise convolution. The depthwise convolution applies spatial filtering independently to each channel, while the pointwise convolution mixes channel information. This reduces computational cost while preserving structured convolutional operations.

### 2.4.3. Low-Rank Decomposition

Low-rank decomposition approximates a dense channel transformation by passing through a smaller intermediate rank. Instead of mapping `Cin` directly to `Cout`, a low-rank 1x1 convolution maps `Cin -> rank -> Cout`. This reduces parameters and computation, especially in channel-mixing layers.

### 2.4.4. Tucker-Style Spatial Decomposition

Tucker-style decomposition approximates a 3x3 convolution using a sequence of smaller operations:

```text
1x1 reduce -> 3x1 convolution -> 1x3 convolution -> 1x1 expand
```

This preserves an effective 3x3 receptive field while reducing parameter cost. It is especially useful in high-channel layers.

### 2.4.5. Post-Training Quantization and Vitis-AI Deployment

Post-training quantization is an optimization technique applied after a model has been trained in floating-point precision. In a typical deployment workflow, the trained FP32 model is exported to an intermediate representation such as ONNX, then quantized to INT8 to reduce memory footprint and improve inference throughput. Quantization is especially important for edge devices because it reduces both model storage and arithmetic cost.

In this project, architectural compression and quantization are complementary rather than mutually exclusive. The custom YOLOv8n model first reduces parameters and GFLOPs through `DSC_LR_Conv`, `LRFuseC2f`, `LRTuckerC2f`, and `DSCDetect`. After that, the compressed model can be exported and quantized for deployment on accelerator platforms such as ZCU102 using Vitis-AI. The expected workflow is:

```text
Train custom YOLOv8n FP32
-> export to ONNX or compatible intermediate graph
-> calibrate and quantize to INT8
-> compile to hardware-specific format
-> deploy on ZCU102 DPU
```

For ZCU102 deployment, the quantized model is expected to be compiled into a DPU-compatible format. This stage is necessary because the FPGA-based accelerator does not directly execute the original PyTorch training model. Instead, it requires a compiled model graph optimized for the target DPU architecture.

| Stage | Purpose | Expected Output |
| --- | --- | --- |
| FP32 training | Learn model parameters with full precision | `.pt` model |
| Export | Convert training model to deployable graph | ONNX or equivalent graph |
| Calibration | Collect activation statistics for quantization | Calibration profile |
| INT8 quantization | Reduce precision for edge acceleration | Quantized model |
| Vitis-AI compilation | Map model to ZCU102 DPU | DPU-compatible compiled model |
| Deployment | Run inference on target board | Real-time detection output |

# CHAPTER 3. ENHANCING YOLOV8N MODEL

## 3.1. Custom YOLOv8n-DSC Model Overview

The proposed model is defined in:

```text
cfg/models/v8/yolov8n_custom_dsc_p45.yaml
```

The model keeps the YOLOv8n scale:

```yaml
n: [0.33, 0.25, 1024]
```

The model is configured for a single-class detection task:

```yaml
nc: 1
reg_max: 8
```

The design principle is to preserve early feature extraction while compressing the high-channel and high-cost parts of the network. Early Conv and C2f layers are kept unchanged. The P4/P5 downsampling layers, selected C2f blocks, and the detection head are replaced by lightweight custom modules.

## 3.2. Model Development Methods

### 3.2.1. DSC_LR_Conv

`DSC_LR_Conv` is used to replace selected standard 3x3 convolution layers. The module consists of:

```text
Depthwise 3x3 convolution
-> Low-rank 1x1 reduce
-> Low-rank 1x1 expand
-> Residual shortcut when possible
```

In the YAML file, `DSC_LR_Conv` is used at layers 5, 7, 16, and 19. These layers are responsible for downsampling transitions at higher feature stages. Replacing them reduces the computational cost of high-channel convolution.

### 3.2.2. LRFuseC2f

`LRFuseC2f` is a modified C2f block that keeps the original C2f input projection and bottleneck structure, but replaces the final fusion convolution with a low-rank 1x1 convolution. This provides a conservative compression method that maintains stability while reducing parameters.

The model uses `LRFuseC2f` at layers 6, 12, and 18.

### 3.2.3. LRTuckerC2f

`LRTuckerC2f` applies stronger compression than `LRFuseC2f`. It replaces both 1x1 projection layers with low-rank versions and uses Tucker-style decomposition inside the bottleneck. This module is used in P5-related blocks, where the number of channels is high and compression is most beneficial.

The model uses `LRTuckerC2f` at layers 8 and 21.

### 3.2.4. DSCDetect

The standard YOLOv8 `Detect` head is replaced by `DSCDetect`. This head keeps the YOLOv8 detection output structure but uses lightweight `DSC_LR_Conv` layers inside the bounding box and classification branches. Since the task has one class and `reg_max=8`, this head further reduces model complexity.

### 3.2.5. Architecture Comparison

The main backbone changes are shown below.

| Layer | YOLOv8n baseline | Custom YOLOv8n |
| ---: | --- | --- |
| 0 | `Conv [64, 3, 2]` | unchanged |
| 1 | `Conv [128, 3, 2]` | unchanged |
| 2 | `C2f [128, True]` | unchanged |
| 3 | `Conv [256, 3, 2]` | unchanged |
| 4 | `C2f [256, True]` | unchanged |
| 5 | `Conv [512, 3, 2]` | `DSC_LR_Conv [512, 3, 2, True, null, 0.125, 8]` |
| 6 | `C2f [512, True]` | `LRFuseC2f [512, True, 1, 0.25, 0.5]` |
| 7 | `Conv [1024, 3, 2]` | `DSC_LR_Conv [1024, 3, 2, True, null, 0.125, 8]` |
| 8 | `C2f [1024, True]` | `LRTuckerC2f [1024, True, 1, 0.5, 0.125]` |
| 9 | `SPPF [1024, 5]` | unchanged |

The main head changes are shown below.

| Layer | YOLOv8n baseline | Custom YOLOv8n |
| ---: | --- | --- |
| 12 | `C2f [512]` | `LRFuseC2f [512, False, 1, 0.5, 0.5]` |
| 15 | `C2f [256]` | unchanged |
| 16 | `Conv [256, 3, 2]` | `DSC_LR_Conv [256, 3, 2, True, null, 0.125, 8]` |
| 18 | `C2f [512]` | `LRFuseC2f [512, False, 1, 0.25, 0.5]` |
| 19 | `Conv [512, 3, 2]` | `DSC_LR_Conv [512, 3, 2, True, null, 0.125, 8]` |
| 21 | `C2f [1024]` | `LRTuckerC2f [1024, False, 1, 0.5, 0.125]` |
| 22 | `Detect [nc]` | `DSCDetect [nc]` |

# CHAPTER 4. IMPLEMENTATION ALGORITHM AND EXPERIMENTAL RESULTS

## 4.1. Training Configuration

Both YOLOv8n baseline and the custom YOLOv8n model are trained for 150 epochs. The evaluation uses 12,655 instances. The comparison includes model size, layer count, parameter count, GFLOPs, precision, recall, mAP, latency, and estimated FPS.

## 4.2. Baseline and Custom Model Results

The main experimental results are summarized below.

| Metric | YOLOv8n baseline | Custom YOLOv8n 676K |
| --- | ---: | ---: |
| Epochs | 150 | 150 |
| Train time (hours) | 2.586 | 2.747 |
| Weight size (MB) | 6.3 | 1.7 |
| Layers | 73 | 152 |
| Parameters | 3,005,843 | 676,811 |
| GFLOPs | 8.1 | 3.2 |
| Instances | 12,655 | 12,655 |
| Precision | 0.878 | 0.881 |
| Recall | 0.875 | 0.856 |
| mAP50 | 0.937 | 0.930 |
| mAP50-95 | 0.668 | 0.651 |
| Preprocess (ms) | 0.2 | 0.2 |
| Inference (ms) | 2.3 | 2.2 |
| Postprocess (ms) | 2.6 | 2.5 |
| Total time/image (ms) | 5.1 | 4.9 |
| Estimated FPS | 196 | 204 |

## 4.3. Comparison and Analysis

Compared with YOLOv8n baseline, the custom model reduces parameters from 3,005,843 to 676,811. This corresponds to a 77.48% parameter reduction and a compression ratio of approximately 4.44x. GFLOPs decrease from 8.1 to 3.2, corresponding to a 60.49% reduction. The weight size decreases from 6.3 MB to 1.7 MB.

The custom model maintains strong detection accuracy. Precision slightly increases from 0.878 to 0.881, while recall decreases from 0.875 to 0.856. The mAP50 decreases from 0.937 to 0.930, meaning the custom model preserves 99.25% of the baseline mAP50. The mAP50-95 decreases from 0.668 to 0.651, showing a small reduction in stricter localization performance.

The latency results show that inference time decreases from 2.3 ms to 2.2 ms, and total processing time per image decreases from 5.1 ms to 4.9 ms. Estimated FPS increases from 196 to 204. This indicates that the custom model improves computational efficiency while maintaining competitive accuracy.

Detailed detection analysis is shown below.

| Metric | YOLOv8n baseline | Custom YOLOv8n 676K |
| --- | ---: | ---: |
| Correct | 10,523 | 10,492 |
| Wrong Pred | 2,516 | 2,501 |
| Missed GT | 1,076 | 1,107 |
| Mean Correct IoU | 0.8467 | 0.8456 |
| Mean Pred Best IoU | 0.7118 | 0.7175 |
| Mean GT Best IoU | 0.7833 | 0.7797 |
| Precision | 0.8070 | 0.8075 |
| Recall | 0.9072 | 0.9046 |
| F1-score | 0.8542 | 0.8533 |

The custom model produces only 31 fewer correct detections than the baseline and slightly fewer wrong predictions. The number of missed ground-truth boxes increases, which explains the recall reduction. Mean Correct IoU remains almost unchanged, showing that the localization quality is preserved when the model detects an object correctly.

# CHAPTER 5. REAL-TIME DEPLOYMENT DISCUSSION

## 5.1. Edge Deployment Requirements

Real-time plant disease detection systems require fast inference, low model size, and stable accuracy. On edge devices, memory capacity and computation are limited. Therefore, reducing parameter count and GFLOPs is important for practical deployment.

The custom YOLOv8n model is designed with these constraints in mind. By replacing high-cost modules with structured lightweight modules, the model becomes more suitable for edge deployment while remaining compatible with standard convolutional runtimes.

## 5.2. Inference Speed and Model Size

The custom model reduces weight size from 6.3 MB to 1.7 MB. This reduction is useful for devices with limited storage or memory bandwidth. The estimated FPS improves from 196 to 204, and total time per image decreases from 5.1 ms to 4.9 ms.

Although the number of layers increases, this is caused by decomposing large convolutional operations into several smaller operations. The final GFLOPs and parameter count are significantly lower than the baseline.

## 5.3. Practical Deployment Considerations

The model should be evaluated on physical edge devices before final deployment. While GFLOPs and parameter count are useful indicators, real latency also depends on memory bandwidth, kernel optimization, runtime framework, and hardware acceleration support.

The custom YOLOv8n model can be further optimized using:

- ONNX export;
- OpenVINO inference;
- INT8 quantization;
- TensorRT acceleration;
- FPGA or DPU deployment.

# CHAPTER 6. HIGH-PERFORMANCE DEPLOYMENT DIRECTION

## 6.1. Hardware Acceleration Motivation

For real-time applications requiring high throughput, hardware acceleration can improve inference performance beyond CPU-only execution. Platforms such as GPUs, NPUs, and FPGA-based accelerators can exploit the structured convolutional operations used in the custom YOLOv8n model.

## 6.2. Quantization and Compilation Direction

The proposed model can be exported to ONNX and then converted to deployment formats such as OpenVINO IR, TensorRT engine, or Vitis-AI xmodel. Post-training quantization can reduce memory footprint and improve inference speed by converting FP32 operations to INT8.

Because the model uses structured convolutional modules rather than irregular pruning, it is more suitable for standard inference compilers and hardware accelerators.

## 6.3. Future Hardware Evaluation

Future work should evaluate the model on target hardware platforms and compare:

- FPS;
- latency per image;
- power consumption;
- memory usage;
- mAP before and after quantization;
- stability under real-time input streams.

# CHAPTER 7. CONCLUSION

This report presents a lightweight custom YOLOv8n model for real-time plant disease detection. The proposed architecture keeps the main YOLOv8n detection topology while replacing selected high-cost modules with `DSC_LR_Conv`, `LRFuseC2f`, `LRTuckerC2f`, and `DSCDetect`.

Experimental results show that the custom model significantly reduces model complexity. Parameters decrease from 3,005,843 to 676,811, GFLOPs decrease from 8.1 to 3.2, and weight size decreases from 6.3 MB to 1.7 MB. The model maintains 0.930 mAP50, preserving 99.25% of the YOLOv8n baseline mAP50, while estimated FPS improves from 196 to 204.

Overall, the custom YOLOv8n model achieves a favorable balance between accuracy and efficiency. It is a practical candidate for edge-oriented plant disease detection and can be further improved through quantization and hardware acceleration.

# REFERENCES

1. Ultralytics YOLOv8 documentation and source code.
2. `cfg/models/v8/yolov8n_custom_dsc_p45.yaml`.
3. `nn/modules/conv.py` for `DSC_LR_Conv`, low-rank convolution, and Tucker-style convolution.
4. `nn/modules/block.py` for `LRFuseC2f` and `LRTuckerC2f`.
5. `nn/modules/head.py` for `DSCDetect`.
6. `baocaolatest.md` for experimental comparison between YOLOv8n baseline and custom YOLOv8n.
7. `BÁO CÁO DATN_LAST_DONE.md` for report structure reference.
