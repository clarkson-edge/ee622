# Transformer Architectures in Biometrics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Advanced applications of transformer architectures in biometric recognition systems, covering fingerprint, face, iris, and voice biometrics with hands-on implementation.

## 📋 Course Overview

This repository contains materials for a graduate-level course exploring how self-attention mechanisms revolutionize biometric feature extraction, representation, and matching. Each week combines theoretical foundations with practical implementations using real datasets like SOCOFing.

**📖 Full Course Details:** See [Syllabus](syllabus.md) for complete information including assessment, grading, and schedule.

## 🚀 Quick Start

### Week 1: Foundational Transformer Architectures
[![Week 1 Colab](https://img.shields.io/badge/Week%201-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%201/lab/week1_transformer_attention_visualization.ipynb)

**Topics:** Transformer fundamentals, self-attention mechanisms, attention visualization
**Lab:** Visualizing attention patterns across biometric modalities (face, fingerprint, iris)

### Week 2: Fingerprint Feature Extraction and Matching
[![Week 2 Colab](https://img.shields.io/badge/Week%202-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%202/lab/week2_nb1_fingerprint_transformer.ipynb)

**Topics:** Hybrid CNN-transformer architectures, quality-aware processing, AFIS integration
**Lab:** SOCOFing dataset processing, core-focused attention analysis, adaptive position encoding

### Week 3: Self-Attention for Minutiae Detection
[![Week 3 Notebook 1](https://img.shields.io/badge/Notebook%201-Traditional-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%203/lab/week3_notebook1_fundamentals.ipynb)
[![Week 3 Notebook 2](https://img.shields.io/badge/Notebook%202-Attention-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%203/lab/week3_notebook2_attention.ipynb)
[![Week 3 Notebook 3](https://img.shields.io/badge/Notebook%203-Production-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%203/lab/week3_notebook3_fixed.ipynb)

**Topics:** Real-world minutiae detection challenges, attention-based detection, privacy-preserving biometrics
**Lab:** Fix "0 detection" problem, implement attention mechanisms, build production system with cancelable templates

### Week 4: Vision Transformers (ViT) for Facial Recognition
[![Week 4 Colab](https://img.shields.io/badge/Week%204-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%204/lab/Week4_FaceBiometrics.ipynb)

**Topics:** Face image patching and sequence processing, ViT architecture adaptation, comparing ViT with CNN approaches
**Lab:** Implementing ViT for face recognition, face verification with transformers, visualization of facial feature attention

### Week 5: Cross-Attention Networks for Facial Attribute Analysis
[![Week 5 Colab](https://img.shields.io/badge/Week%205-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%205/lab/Week5_Notebook1.ipynb)

**Topics:** Cross-attention mechanisms for multi-attribute learning, handling extreme class imbalance, attention visualization
**Lab:** Complete implementation from dataset imbalance discovery to advanced visualization, focal loss and aggressive weighting strategies

## 📚 Repository Structure

```
├── Week 1/
│   ├── lab/
│   │   └── week1_transformer_attention_visualization.ipynb
│   ├── materials/
│   │   ├── Iris-SAM.pdf
│   │   └── NIPS - Attention is All You Need.pdf
│   ├── notes/
│   │   ├── EE622 Chapter 1.md
│   │   ├── EE622 Chapter 1.odt
│   │   └── EE622 Chapter 1.pdf
│   ├── slides/
│   │   ├── ViT-Architecture.png
│   │   └── week1_transformer_biometrics_presentation.pptx
│   └── README.md
├── Week 2/
│   ├── lab/
│   │   └── week2_nb1_fingerprint_transformer.ipynb
│   ├── notes/
│   │   ├── EE622 Chapter 2.md
│   │   ├── EE622 Chapter 2.odt
│   │   └── EE622 Chapter 2.pdf
│   ├── slides/
│   │   └── week2_theory_fingerprint_transformers.pptx
│   └── README.md
├── Week 3/
│   ├── lab/
│   │   ├── week3_notebook1_fundamentals.ipynb
│   │   ├── week3_notebook2_attention.ipynb
│   │   └── week3_notebook3_fixed.ipynb
│   ├── notes/
│   │   ├── EE622 Chapter 3.md
│   │   ├── EE622 Chapter 3.odt
│   │   └── EE622 Chapter 3.pdf
│   └── README.md
├── Week 4/
│   ├── lab/
│   │   └── Week4_FaceBiometrics.ipynb
│   ├── notes/
│   │   ├── EE622 Chapter 4.md
│   │   ├── EE622 Chapter 4.odt
│   │   └── EE622 Chapter 4.pdf
│   ├── slides/
│   │   └── Week4_FaceBiometrics.pptx
│   ├── facial-recognition-glossary.md
│   └── README.md
├── Week 5/
│   ├── lab/
│   │   └── Week5_Notebook1.ipynb
│   └── README.md
├── Week 6-10/
│   └── (Under development)
├── syllabus.md
├── biometric_transformer_cheatsheet.md
├── biometrics-glossary.md
├── transformer-formulas-reference.md
├── graduate-projects.md
└── weekly-theory-practice-guide.md
```

## 🎯 Learning Objectives

- Master transformer architectures for biometric applications
- Implement self-attention mechanisms for feature extraction and matching
- Design hybrid CNN-transformer systems with quality-aware processing
- Visualize and interpret attention patterns (core-focused analysis)
- Evaluate performance against traditional methods using standard datasets
- Develop adaptive position encoding for biometric-specific spatial relationships
- Build production-ready biometric systems with privacy preservation

## 🛠️ Setup

### Google Colab (Recommended)
Click the Colab badges above to open notebooks directly from the GitHub repository. The notebooks include:
- SOCOFing dataset access via kagglehub
- Pre-trained transformer models
- Visualization libraries for attention analysis
- Privacy-preserving template generation

### Local Development
```bash
# Create environment
conda create -n biometric-transformers python=3.8+
conda activate biometric-transformers

# Install packages
pip install torch torchvision transformers
pip install opencv-python scikit-image matplotlib seaborn
pip install kagglehub gradio numpy scipy pandas
pip install einops  # For attention mechanisms (Week 3)
pip install plotly ipywidgets imageio  # For advanced visualizations (Week 5)
pip install face-recognition dlib timm  # For face biometrics (Week 4)
```

## 📖 Key Resources & References

### 📋 Essential Reference Materials
- **🔧 Implementation Guide:** [Biometric Transformer Architecture Cheat Sheet](biometric_transformer_cheatsheet.md)
  - Complete implementation patterns for all biometric modalities
  - SOCOFing dataset integration examples
  - Quality-aware attention mechanisms
  - Cross-modal fusion strategies

- **📚 Technical Glossary:** [Biometric Transformers Key Terms](biometrics-glossary.md)
  - Comprehensive terminology for transformer architectures
  - Fingerprint-specific concepts (Poincaré index, ridge coherence, core detection)
  - Cross-modal and multimodal fusion definitions
  - Implementation and evaluation concepts

- **🧮 Mathematical Foundations:** [Transformer Formulas Reference](transformer-formulas-reference.md)
  - Core transformer equations with biometric adaptations
  - Fingerprint analysis formulas (orientation estimation, quality assessment)
  - Performance metrics and evaluation mathematics
  - Training objectives and optimization techniques

- **🎓 Research Projects:** [Graduate Project Ideas](graduate-projects.md)
  - 24 moderate-difficulty projects for 45-day completion
  - Covers all major biometric modalities
  - Includes evaluation criteria and deliverables

### 📄 Foundational Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) - Original transformer architecture
- [Vision Transformers](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020) - ViT foundation
- [Transformer Fingerprint Recognition](https://ieeexplore.ieee.org/document/9956435) (IEEE 2022) - Biometric applications
- [Cross-Spectral Vision Transformer](https://arxiv.org/html/2412.19160v2) (2024) - Advanced biometric fusion

### 🌐 External Resources
- [Course GitHub Repository](https://github.com/clarkson-edge/ee622) - Complete course materials and notebooks
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanations
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Implementation library
- [Papers With Code](https://paperswithcode.com/methods/category/transformers) - Latest research
- [SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing) - Real fingerprint data

## 🧪 What You'll Build

### Week 1: Multi-Modal Attention Visualization System
- Extract attention maps from pre-trained Vision Transformers
- Visualize attention patterns across biometric modalities (face, fingerprint, iris)
- Compare attention evolution across transformer layers
- Identify biometrically significant regions through attention analysis

### Week 2: Advanced Fingerprint Recognition Pipeline
- **SOCOFing Dataset Integration:** Real African fingerprint data processing
- **Adaptive Preprocessing:** Gabor filtering, orientation estimation, core detection using Poincaré index
- **Quality-Aware Transformers:** Patch quality assessment and attention weighting
- **Core-Focused Analysis:** Attention correlation with fingerprint structure
- **Hybrid Architecture:** CNN backbone + transformer layers for global-local features

### Week 3: Production Minutiae Detection System
- **Real-World Problem Solving:** Fix traditional detection failing on real data (0→24+ minutiae)
- **Attention-Based Detection:** Type-specific attention heads for endings and bifurcations
- **Ensemble Methods:** Combine traditional and attention approaches for robustness
- **Privacy-Preserving Templates:** Cancelable biometrics with one-way transformations
- **Production API:** Complete system with error handling and monitoring

### Week 4: Vision Transformer Face Recognition
- **ViT Implementation:** Build Vision Transformer from scratch for faces
- **Face Preprocessing:** Detection, alignment, and normalization pipeline
- **Attention Analysis:** Visualize which facial features matter most
- **Performance Comparison:** Benchmark against FaceNet and ArcFace
- **Multi-Scale Processing:** Handle varying face sizes and poses

### Week 5: Cross-Attention for Facial Attributes
- **Extreme Imbalance Handling:** Solve 2% positive rate challenges
- **Focal Loss Implementation:** Advanced loss functions for rare attributes
- **Interactive Visualization:** Build attention explorers with frequency awareness
- **Production Solutions:** From stuck models to successful deployment
- **Comprehensive Analysis:** ROC curves, F1 scores, and performance dashboards

### Key Innovations Demonstrated
- **Adaptive Position Encoding:** Fingerprint core detection and relative spatial encoding
- **Quality-Aware Attention:** Dynamic weighting based on patch reliability
- **Multi-Layer Attention Analysis:** Evolution from structural to semantic focus
- **Extreme Imbalance Solutions:** Focal loss, 50x weights, diversity penalties
- **Production-Ready Systems:** APIs, privacy preservation, and monitoring