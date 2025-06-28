# [ICCV2025] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation
### Abstract
In the effort to achieve robust and generalizable category-level object pose estimation, recent methods primarily focus on learning fundamental representations from data. However, the inherent biases within the data are often overlooked: the repeated training samples and similar environments may mislead the models to over-rely on specific patterns, hindering models' performance on novel instances. In this paper, we present CleanPose, a novel method that mitigates the data biases to enhance category-level pose estimation by integrating causal learning and knowledge distillation. By incorporating key causal variables (structural information and hidden confounders) into causal modeling, we propose the causal inference module based on front-door adjustment, which promotes unbiased estimation by reducing potential spurious correlations. Additionally, to further confront the data bias at the feature level, we devise a residual-based knowledge distillation approach to transfer unbiased semantic knowledge from 3D foundation model, providing comprehensive causal supervision. Extensive experiments across multiple benchmarks (REAL275, CAMERA25 and HouseCat6D) hightlight the superiority of proposed CleanPose over state-of-the-art methods.

![](/main_v3.png)

[[paper](https://arxiv.org/pdf/2502.01312)]

## Code is being organized. Please wait patiently~
