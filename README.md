# DIAMOND

**DIAMOND** is a novel deep learning framework for medical image segmentation that integrates a dual-encoder hybrid backbone and a quasi-multimodal training paradigm. It achieves robust generalization, computational efficiency, and strong adaptability across heterogeneous datasets.
![Diamond](https://github.com/user-attachments/assets/dd6a39c8-ccc0-49c1-8784-a4cf4083ef0a)


---

## Features

- **Dual-Encoder Hybrid Backbone**: Combines a convolutional encoder (for local feature extraction) with a Swin-Transformer encoder (for global context modeling).
- **Residual Recursive Gated Convolution (RrgConv)**: A lightweight, portable module that stabilizes high-order spatial interactions with low computational overhead.
- **Nested Attention System (NAS)**: Integrates spatial and channel attention within self-attention to enhance feature representation and focus.
- **Quasi-Multimodal Training (QMM)**: Facilitates training across multiple datasets focused on the same lesion category, reducing annotation heterogeneity and improving cross-dataset generalization.
- **Comprehensive Evaluation**: Benchmarked on 10 public datasets and 33 state-of-the-art models, showing competitive or superior performance with reduced resource demands.

---

## Repository Structure

```text
├── LICENSE
├── README.md
├── Diamond/
│   ├── Diamond_ECDC.py          # Diamond Backbone Network CNN encoder – CNN decoder
│   ├── Diamond_ECDT.py          # Diamond Backbone Network CNN encoder – Transformer decoder
│   ├── Diamond_ETDC.py          # Diamond Backbone Network Transformer encoder – CNN decoder
│   ├── Diamond_ETDT.py          # Diamond Backbone Network Transformer encoder – Transformer decoder
│   ├── Diamond_ECDC_NAS.py      # Diamond_ECDC with MHSA replaced by Nested Attention System
│   ├── Diamond_ECDC_Rrg.py      # Diamond_ECDC with Double Conv replaced by RrgConv in Bottleneck
│   ├── ...
│   ├── DoubleConv.py            # Double Convolution Module
│   ├── DWSCov.py                # Depthwise Separable Convolution
│   ├── HorBlock.py              # The HorBlock of the HorNet
│   ├── MHSA.py                  # Multi Head Self Attention
│   ├── NAS.py                   # Nested Attention System
│   ├── RrgConv.py               # Residual Recursive Gated Convolution
│   ├── Transition.py            # 1x1 Convolution Transition Module
│   └── ...
├── Portability/
│   ├── DoubleConv.py            # Double Convolution Module
│   ├── DWSCov.py                # Depthwise Separable Convolution
│   ├── HorBlock.py              # The HorBlock of the HorNet
│   ├── NAS.py                   # Nested Attention System
│   ├── RrgConv.py               # Residual Recursive Gated Convolution
│   ├── U_Net_NAS.py             # U-Net with Nested Attention System inserted in Bottleneck
│   └── U-Net_Rrg.py             # U-Net with Double Conv replaced by RrgConv in Bottleneck
```
## Results & Benchmarks

- DIAMOND achieves superior Dice and IoU metrics on multiple datasets.
- The QMM strategy consistently improves cross-dataset generalization.
- Lower computational and memory requirements compared to many existing models.

Sample qualitative results are provided in the `results/sample_outputs/` directory.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
