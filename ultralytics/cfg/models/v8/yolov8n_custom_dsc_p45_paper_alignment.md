# Progressive Compressing YOLOv8n by 4.44x while Maintaining 99.25% mAP50 for Real-Time UAV-Based Plant Disease Detection: A Systematic Approach for Edge Deployment

## ABSTRACT

Deploying object detection models on edge devices remains challenging because real-time agricultural monitoring systems must balance detection accuracy, model size, computational cost, and inference latency. Standard YOLO models provide strong detection performance, but their convolutional layers and multi-scale detection heads still require substantial parameters and floating-point operations when deployed on embedded platforms. This paper presents a systematic, hardware-aware compression approach for YOLOv8n in UAV-based plant disease detection. The proposed design progressively compresses high-cost regions of the network while preserving the original YOLOv8n detection topology. Specifically, high-channel downsampling layers are replaced with depthwise separable convolution combined with low-rank pointwise decomposition, selected C2f blocks are replaced with low-rank fusion or Tucker-style C2f variants, and the detection head is redesigned using lightweight DSC-LR convolution branches.

The proposed model is evaluated against a YOLOv8n baseline under the same 150-epoch training setting and 12,655 evaluated instances. Compared with the baseline, the custom YOLOv8n reduces parameters from 3,005,843 to 676,811, corresponding to a 4.44x compression ratio and a 77.48% parameter reduction. GFLOPs decrease from 8.1 to 3.2, while model weight size decreases from 6.3 MB to 1.7 MB. Despite this compression, the model maintains competitive detection accuracy, achieving 0.930 mAP50 compared with 0.937 for the baseline, preserving 99.25% of the original mAP50. The custom model also improves estimated FPS from 196 to 204. These results show that structured progressive compression with depthwise separable and low-rank operations provides a practical pathway for adapting YOLOv8n to real-time, resource-constrained plant disease detection.

## INDEX TERMS

YOLOv8n, object detection, plant disease detection, UAV imagery, edge computing, depthwise separable convolution, low-rank decomposition, Tucker decomposition, model compression, real-time inference.

## I. INTRODUCTION

The rapid development of smart agriculture has created an increasing demand for automated systems that can detect plant diseases quickly and accurately over large areas. UAV imagery provides a practical way to monitor farms and forest regions at scale, but it also introduces difficult visual conditions: targets can be small, densely distributed, and visually similar to the surrounding background. For such scenarios, real-time object detection is essential because delayed detection may reduce the value of early intervention.

YOLO-based detectors are widely used in real-time applications because they perform localization and classification in a single forward pass. YOLOv8n, in particular, is a compact member of the YOLOv8 family and is suitable as a baseline for edge-oriented detection. However, even compact detectors can remain expensive for embedded deployment. Convolutional layers with high channel counts, repeated feature-fusion blocks, and detection heads operating at multiple feature scales contribute to parameter count, memory traffic, and inference cost.

A key challenge is therefore how to reduce the model cost without destroying the detection capability required for small and dense disease regions. Unstructured pruning can reduce parameter count but often creates irregular computation patterns that are difficult to accelerate on commodity hardware. In contrast, structured compression techniques such as depthwise separable convolution and low-rank decomposition preserve regular tensor operations and are more suitable for edge deployment. This motivation follows the design philosophy of the paper "Progressive Compressing ResNet by 8.55 while Maintaining 89.5% Accuracy on ImageNet", which argues that hardware-friendly compression should use structured operations, preserve useful residual pathways, and allow a controllable accuracy-compression trade-off.

Inspired by that framework, this paper adapts progressive structured compression to YOLOv8n. Instead of redesigning the entire detector from scratch, the proposed method keeps the early feature extraction path and YOLOv8n multi-scale detection topology, then compresses the higher-cost P4/P5 regions and the detection head. This produces a lighter model while retaining the main architectural advantages of YOLOv8n for small-object detection.

The main contributions of this work are as follows:

1. A compressed YOLOv8n architecture is proposed for UAV-based plant disease detection, using depthwise separable convolution and low-rank decomposition in high-cost network regions.
2. A progressive replacement strategy is applied: early P1/P2/P3 layers remain unchanged, while P4/P5 downsampling layers, selected C2f blocks, and the detection head are compressed.
3. Experimental results show that the proposed model achieves a 4.44x parameter compression ratio and reduces GFLOPs by 60.49%, while preserving 99.25% of the baseline mAP50.

## II. RELATED WORK

### A. YOLO-Based Real-Time Object Detection

YOLO detectors formulate object detection as a single-stage regression problem, enabling fast inference compared with multi-stage approaches. YOLOv8 improves the modularity of earlier YOLO designs through a convolutional backbone, C2f feature blocks, feature pyramid and path aggregation structures, and a decoupled detection head. YOLOv8n is the smallest standard version and is often selected when real-time deployment is important.

Although YOLOv8n is lightweight compared with larger variants, its cost is still concentrated in repeated convolutional operations. High-channel 3x3 convolutions and detection branches are especially expensive because they process feature maps at multiple pyramid levels. For edge devices, reducing these layers is important not only for parameter storage but also for memory movement and latency.

### B. Depthwise Separable Convolution

Depthwise separable convolution decomposes a standard convolution into two stages: a depthwise spatial convolution that processes each input channel independently, followed by a pointwise 1x1 convolution that mixes channel information. This reduces the spatial convolution cost while retaining a learnable channel-mixing stage. The paper used as reference emphasizes this operation because it produces regular dense computation rather than irregular sparse patterns.

In the proposed YOLOv8n variant, depthwise separable convolution is used inside `DSC_LR_Conv`. This module replaces selected high-channel 3x3 convolution layers, especially those responsible for downsampling at P4/P5 and in the detection head.

### C. Low-Rank Channel Decomposition

Low-rank decomposition reduces a dense channel transformation by factorizing it through a lower-dimensional bottleneck. A standard 1x1 convolution maps `Cin -> Cout` directly, while a low-rank alternative maps `Cin -> r -> Cout`, where `r` is the rank. This provides a continuous compression knob: smaller rank reduces parameters and computation, while larger rank preserves more representational capacity.

Following the same principle, the proposed model uses low-rank 1x1 convolution in `DSC_LR_Conv`, `LRFuseC2f`, and `LRTuckerC2f`. The rank is controlled by `rank_ratio`, allowing the compression strength to be adjusted systematically.

### D. Structured Compression for Edge Deployment

For edge deployment, compression should reduce model size and computation while keeping operations compatible with existing deep-learning runtimes. Structured compression is preferable because it retains regular convolutional operators and predictable tensor shapes. The reference paper demonstrates this idea on ResNet by progressively introducing depthwise separable convolution and low-rank decomposition. This work transfers the same idea to YOLOv8n detection: the original detector topology is preserved, but expensive components are replaced with structured lightweight equivalents.

## III. METHODOLOGY

The proposed method compresses YOLOv8n progressively and selectively. The model is defined in:

```text
cfg/models/v8/yolov8n_custom_dsc_p45.yaml
```

The model keeps YOLOv8n compound scaling:

```yaml
n: [0.33, 0.25, 1024]
```

The detection task uses one class:

```yaml
nc: 1
reg_max: 8
```

The compression is applied mainly to P4/P5 and the head downsampling path, while the early P1/P2/P3 feature extraction path is kept unchanged.

### A. Progressive Compression Strategy

The design follows a stage-wise compression strategy:

1. Keep early layers unchanged to preserve low-level visual features and small-object sensitivity.
2. Replace high-channel downsampling convolutions in P4/P5 with `DSC_LR_Conv`.
3. Replace selected C2f blocks with low-rank variants depending on the desired compression strength.
4. Replace the standard `Detect` head with `DSCDetect` to reduce the cost of classification and bounding-box branches.

This strategy avoids compressing every layer uniformly. Instead, it targets the most expensive regions while preserving important detection pathways.

### B. Depthwise Separable Low-Rank Convolution

`DSC_LR_Conv` is the main lightweight convolution used in the proposed model. It replaces a standard 3x3 convolution with:

```text
Depthwise 3x3 convolution
-> Low-rank 1x1 reduce
-> Low-rank 1x1 expand
-> Residual shortcut when stride = 1 and Cin = Cout
```

The YAML configuration uses:

```yaml
DSC_LR_Conv, [Cout, 3, 2, True, null, 0.125, 8]
```

Here, `rank_ratio=0.125` sets the automatic rank to 12.5% of the smaller channel dimension, while `rank_min=8` prevents the rank from becoming too small. This keeps the operation compact but avoids an excessively narrow bottleneck.

### C. Low-Rank Fusion C2f

`LRFuseC2f` is a conservative C2f variant. It keeps the original `cv1` and bottleneck blocks, but replaces the final fusion convolution `cv2` with a low-rank 1x1 convolution:

```text
Input
-> original C2f cv1
-> original Bottleneck blocks
-> concatenation
-> low-rank 1x1 fusion
-> output
```

This module is used where stability is important and full decomposition may be too aggressive. In the YAML file, `LRFuseC2f` appears at layers 6, 12, and 18.

### D. Tucker-Style Low-Rank C2f

`LRTuckerC2f` applies stronger compression than `LRFuseC2f`. It replaces both 1x1 projections with low-rank versions and replaces internal 3x3 convolution with a Tucker-style decomposition:

```text
1x1 reduce -> 3x1 convolution -> 1x3 convolution -> 1x1 expand
```

This preserves an effective 3x3 receptive field while reducing the parameter cost. The module is applied at P5, where channel count is highest and compression brings the largest benefit.

### E. Lightweight Detection Head

The standard YOLOv8 `Detect` head is replaced by `DSCDetect`. It keeps the decoding behavior of YOLOv8 but changes the internal box and class branches:

```text
box branch: DSC_LR_Conv -> DSC_LR_Conv -> output convolution
class branch: DSC_LR_Conv -> DSC_LR_Conv -> output convolution
```

Since the target task has one class and `reg_max=8`, this produces a compact detection head while preserving the same three-scale output structure: P3, P4, and P5.

### F. Architecture-Level Changes

Backbone changes are summarized below.

| Layer | YOLOv8n baseline | Proposed model | Purpose |
| ---: | --- | --- | --- |
| 0 | `Conv [64, 3, 2]` | unchanged | Preserve early edge and texture features |
| 1 | `Conv [128, 3, 2]` | unchanged | Keep stable low-level extraction |
| 2 | `C2f [128, True]` | unchanged | Preserve local feature representation |
| 3 | `Conv [256, 3, 2]` | unchanged | Preserve P3 small-object information |
| 4 | `C2f [256, True]` | unchanged | Preserve P3/8 feature quality |
| 5 | `Conv [512, 3, 2]` | `DSC_LR_Conv [512, 3, 2, True, null, 0.125, 8]` | Compress P4 downsampling |
| 6 | `C2f [512, True]` | `LRFuseC2f [512, True, 1, 0.25, 0.5]` | Low-rank fusion at P4 |
| 7 | `Conv [1024, 3, 2]` | `DSC_LR_Conv [1024, 3, 2, True, null, 0.125, 8]` | Compress P5 downsampling |
| 8 | `C2f [1024, True]` | `LRTuckerC2f [1024, True, 1, 0.5, 0.125]` | Strong P5 compression |
| 9 | `SPPF [1024, 5]` | unchanged | Preserve receptive-field aggregation |

Head changes are summarized below.

| Layer | YOLOv8n baseline | Proposed model | Purpose |
| ---: | --- | --- | --- |
| 10 | Upsample | unchanged | Parameter-free operation |
| 11 | Concat P4 | unchanged | Preserve feature pyramid topology |
| 12 | `C2f [512]` | `LRFuseC2f [512, False, 1, 0.5, 0.5]` | Compress fusion after P5/P4 concat |
| 13 | Upsample | unchanged | Parameter-free operation |
| 14 | Concat P3 | unchanged | Preserve small-object pathway |
| 15 | `C2f [256]` | unchanged | Preserve P3 detection quality |
| 16 | `Conv [256, 3, 2]` | `DSC_LR_Conv [256, 3, 2, True, null, 0.125, 8]` | Compress P3-to-P4 transition |
| 17 | Concat P4 | unchanged | Preserve PAN aggregation |
| 18 | `C2f [512]` | `LRFuseC2f [512, False, 1, 0.25, 0.5]` | Compress P4 fusion |
| 19 | `Conv [512, 3, 2]` | `DSC_LR_Conv [512, 3, 2, True, null, 0.125, 8]` | Compress P4-to-P5 transition |
| 20 | Concat P5 | unchanged | Preserve connection to SPPF features |
| 21 | `C2f [1024]` | `LRTuckerC2f [1024, False, 1, 0.5, 0.125]` | Strong P5 head compression |
| 22 | `Detect [nc]` | `DSCDetect [nc]` | Lightweight detection branches |

## IV. EXPERIMENTAL RESULTS

### A. Experimental Setup

The proposed model is compared with a YOLOv8n baseline using the same training duration and evaluation setting. Both models are trained for 150 epochs and evaluated on 12,655 instances. The comparison focuses on model size, computational cost, latency, FPS, and detection accuracy.

### B. Compression and Efficiency

| Metric | YOLOv8n baseline | Proposed YOLO custom 676K | Change |
| --- | ---: | ---: | ---: |
| Epochs | 150 | 150 | 0 |
| Train time (hours) | 2.586 | 2.747 | +6.23% |
| Weight size (MB) | 6.3 | 1.7 | -73.02% |
| Layers | 73 | 152 | +108.22% |
| Parameters | 3,005,843 | 676,811 | -77.48% |
| Compression ratio | 1.00x | 4.44x | +4.44x |
| GFLOPs | 8.1 | 3.2 | -60.49% |
| Preprocess (ms) | 0.2 | 0.2 | 0.00% |
| Inference (ms) | 2.3 | 2.2 | -4.35% |
| Postprocess (ms) | 2.6 | 2.5 | -3.85% |
| Total time/image (ms) | 5.1 | 4.9 | -3.92% |
| Estimated FPS | 196 | 204 | +4.08% |

The proposed model achieves a substantial reduction in model complexity. The number of parameters decreases from 3,005,843 to 676,811, equivalent to a 4.44x compression ratio. GFLOPs decrease by 60.49%, and the weight file size decreases by 73.02%. Although the number of layers increases because large convolutions are decomposed into several smaller operations, the total computational cost is much lower.

The measured inference time decreases from 2.3 ms to 2.2 ms, and the total processing time per image decreases from 5.1 ms to 4.9 ms. The estimated FPS increases from 196 to 204. This shows that structured compression can improve deployment efficiency while keeping the model compatible with standard convolutional operators.

### C. Detection Accuracy

| Metric | YOLOv8n baseline | Proposed YOLO custom 676K | Change |
| --- | ---: | ---: | ---: |
| Precision | 0.878 | 0.881 | +0.003 |
| Recall | 0.875 | 0.856 | -0.019 |
| mAP50 | 0.937 | 0.930 | -0.007 |
| mAP50-95 | 0.668 | 0.651 | -0.017 |
| Retained mAP50 | 100.00% | 99.25% | -0.75% |
| Retained mAP50-95 | 100.00% | 97.46% | -2.54% |

The proposed model maintains competitive detection accuracy despite the large reduction in parameters and GFLOPs. Precision slightly improves from 0.878 to 0.881, indicating that the compressed model does not introduce more false positive detections. Recall decreases from 0.875 to 0.856, suggesting that some difficult or small instances are missed after compression. However, mAP50 only decreases by 0.007, meaning the proposed model retains 99.25% of the baseline mAP50.

The mAP50-95 metric decreases from 0.668 to 0.651. This indicates a mild reduction in stricter localization accuracy, which is expected because low-rank bottlenecks reduce representational capacity. Nevertheless, the accuracy drop is small relative to the 4.44x parameter compression.

### D. Detection Error and IoU Analysis

| Metric | YOLOv8n baseline | Proposed YOLO custom 676K |
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

The detailed detection analysis supports the same conclusion. The proposed model produces 10,492 correct detections, only 31 fewer than the baseline. Wrong predictions decrease slightly from 2,516 to 2,501. The main cost of compression is an increase in missed ground-truth objects from 1,076 to 1,107, which explains the recall reduction.

Localization quality remains nearly unchanged. Mean Correct IoU decreases only from 0.8467 to 0.8456, while Mean Pred Best IoU improves from 0.7118 to 0.7175. This suggests that when the compressed model detects an object, its bounding-box quality remains close to the baseline.

### E. Discussion

The results demonstrate a favorable accuracy-efficiency trade-off. The proposed model reduces the parameter count by 77.48% and GFLOPs by 60.49%, while mAP50 decreases by only 0.75% relative. This is consistent with the central idea of progressive structured compression: large savings are possible when compression is applied to high-cost layers while preserving important feature pathways.

The increase in layer count is an important deployment consideration. Low-rank and Tucker-style decomposition split one large convolution into multiple smaller convolutions. This reduces theoretical computation and parameter count, but real hardware speed depends on kernel launch overhead, memory bandwidth, and runtime optimization. Therefore, future deployment should evaluate the model directly on the intended edge hardware.

Overall, the proposed architecture is especially suitable when memory footprint and model size are more constrained than absolute peak accuracy. It preserves nearly all baseline mAP50 while producing a much smaller model, making it a practical candidate for edge-based plant disease monitoring.

## V. CONCLUSION

This paper presents a progressive structured compression approach for YOLOv8n in UAV-based plant disease detection. Inspired by hardware-aware compression principles, the proposed model combines depthwise separable convolution, low-rank pointwise decomposition, Tucker-style spatial decomposition, and a lightweight detection head. Instead of compressing the entire network, the method targets high-channel P4/P5 stages and the detection head while preserving early feature extraction layers and the YOLOv8n multi-scale detection topology.

Experimental results show that the proposed model reduces parameters from 3,005,843 to 676,811, achieving a 4.44x compression ratio. GFLOPs decrease from 8.1 to 3.2, and weight size decreases from 6.3 MB to 1.7 MB. At the same time, the model maintains 0.930 mAP50, preserving 99.25% of baseline mAP50, and increases estimated FPS from 196 to 204. These findings indicate that structured compression is an effective approach for adapting YOLOv8n to real-time edge deployment while maintaining strong detection performance.

Future work should evaluate this compressed model on physical edge platforms, combine it with INT8 quantization, and study different `rank_ratio` settings to further explore the trade-off between accuracy, latency, and energy consumption.

## REFERENCES AND INTERNAL SOURCES

| Source | Usage |
| --- | --- |
| `Progressive_Compressing_ResNet_by_8_55__while_Maintaining_89_5_Accuracy_on_ImageNet__A_Systematic_Approach_for_Edge_Deployment (1).md` | Article structure and compression motivation |
| `baocaolatest.md` | Experimental metrics for YOLOv8n baseline and proposed custom model |
| `cfg/models/v8/yolov8n_custom_dsc_p45.yaml` | Proposed model architecture |
| `nn/modules/conv.py` | `DSC_LR_Conv`, low-rank convolution, Tucker-style convolution definitions |
| `nn/modules/block.py` | `LRFuseC2f` and `LRTuckerC2f` definitions |
| `nn/modules/head.py` | `DSCDetect` definition |
