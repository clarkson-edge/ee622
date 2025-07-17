# Transformer Architectures in Biometrics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Advanced applications of transformer architectures in biometric recognition systems, covering fingerprint, face, iris, voice, ECG, and multimodal biometrics with hands-on implementation.

## 📋 Course Overview

This repository contains materials for a graduate-level course exploring how self-attention mechanisms revolutionize biometric feature extraction, representation, and matching. Each week combines theoretical foundations with practical implementations using real datasets like SOCOFing, CelebA, ASVspoof, and PhysioNet ECG-ID.

**📖 Full Course Details:** See [Syllabus](syllabus.md) for complete information including assessment, grading, and schedule.

## 🚀 Quick Start

### Week 1: Foundational Transformer Architectures
[![Week 1 Colab](https://img.shields.io/badge/Week%201-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%201/lab/week1_transformer_attention_visualization.ipynb)

**Topics:** Transformer fundamentals, self-attention mechanisms, attention visualization
- Evolution from RNNs/CNNs to transformers
- Self-attention mechanism: Query, Key, Value concepts
- Vision Transformers (ViT) for biometric images
- **Lab:** Visualizing attention patterns across biometric modalities (face, fingerprint, iris)

### Week 2: Fingerprint Feature Extraction and Matching
[![Week 2 Colab](https://img.shields.io/badge/Week%202-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%202/lab/week2_nb1_fingerprint_transformer.ipynb)

**Topics:** Hybrid CNN-transformer architectures, quality-aware processing, SOCOFing dataset
- Advanced preprocessing: Gabor filtering, orientation estimation
- Core detection using Poincaré index
- Quality-aware attention mechanisms
- **Lab:** Complete fingerprint transformer with core-focused attention analysis

### Week 3: Self-Attention for Minutiae Detection
[![Week 3 Notebook 1](https://img.shields.io/badge/Notebook%201-Traditional-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%203/lab/week3_notebook1_fundamentals.ipynb)
[![Week 3 Notebook 2](https://img.shields.io/badge/Notebook%202-Attention-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%203/lab/week3_notebook2_attention.ipynb)
[![Week 3 Notebook 3](https://img.shields.io/badge/Notebook%203-Production-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%203/lab/week3_notebook3_fixed.ipynb)

**Topics:** Real-world minutiae detection challenges, attention-based detection, privacy-preserving biometrics
- Debugging "0 detection" problem with adaptive binarization
- Type-specific attention heads for ridge endings and bifurcations
- Cancelable biometric templates for privacy
- **Lab:** Three-notebook journey from problem discovery to production system

### Week 4: Vision Transformers (ViT) for Facial Recognition
[![Week 4 Colab](https://img.shields.io/badge/Week%204-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%204/lab/Week4_FaceBiometrics.ipynb)

**Topics:** ViT architecture adaptation for face biometrics
- Face image patching strategies (16×16 patches)
- Face-specific position encoding
- Comparison with FaceNet and ArcFace
- **Lab:** Complete ViT implementation for face verification and identification

### Week 5: Cross-Attention Networks for Facial Attribute Analysis
[![Week 5 Colab](https://img.shields.io/badge/Week%205-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%205/lab/Week5_Notebook1.ipynb)

**Topics:** Multi-attribute learning with extreme class imbalance
- Cross-attention mechanisms for attribute-image relationships
- Handling 2% positive rate attributes (e.g., Bald, Mustache)
- Focal loss and aggressive weighting strategies
- **Lab:** Journey from all-negative predictions to successful multi-attribute classification

### Week 6: Contactless Biometric Fusion
[![Week 6 Colab](https://img.shields.io/badge/Week%206-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%206/lab/Week6_FusionModel.ipynb)

**Topics:** Quality assessment and fusion for contactless fingerprint/palmprint
- Traditional quality assessment (LQA_S and GQA_L)
- Two-stage fusion strategy with quality weighting
- Transformer-based quality assessment
- **Lab:** Complete fusion system with synthetic data generation using StyleGAN2

### Week 7: Gait Recognition with Spatial-Temporal Transformers
🚧 **Under Development** - Coming Soon

**Topics:** Analyzing human gait patterns for identification at a distance
- Spatial-temporal transformer architectures
- Gait cycle analysis and feature extraction
- Cross-view gait recognition

### Week 8: Audio Transformers for Speaker Verification with Anti-Spoofing
[![Week 8 Colab](https://img.shields.io/badge/Week%208-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%208/lab/Week8_AudioTransformers.ipynb)

**Topics:** Dual-task transformers for speaker verification and liveness detection
- ASVspoof 2021 dataset integration
- Multi-objective training: speaker discrimination + anti-spoofing
- Attention to temporal inconsistencies in synthetic speech
- **Lab:** Production speaker verification system with integrated spoofing detection

### Week 9: ECG Transformers for Biometric Authentication
[![Week 9 Colab](https://img.shields.io/badge/Week%209-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%209/lab/Week9_ECGAuthentication.ipynb)

**Topics:** Physiological biometrics using cardiac signals
- PhysioNet ECG-ID dataset processing
- Heartbeat segmentation and sequence creation
- Transformer architecture for ECG patterns
- **Lab:** Complete ECG authentication system with real-time processing capabilities

### Week 10: Multimodal Biometric Fusion with Cross-Attention
🚧 **Under Development** - Coming Soon

**Topics:** Combining multiple biometric modalities
- Cross-attention for multimodal fusion
- Score-level and feature-level fusion strategies
- Handling missing modalities

## 📚 Course Resources

### Required Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- See individual week READMEs for specific dependencies

### Datasets Used
- **SOCOFing**: African fingerprint dataset (Weeks 2-3)
- **CelebA**: Facial attributes dataset (Week 5)
- **LFW**: Labeled Faces in the Wild (Week 4)
- **ASVspoof 2021**: Voice anti-spoofing (Week 8)
- **PhysioNet ECG-ID**: ECG biometrics (Week 9)

### Reference Materials
- [Biometric Transformer Cheatsheet](biometric_transformer_cheatsheet.md)
- [Course Bibliography](bibliography.md) (if available)
- Individual chapter notes in each week's folder

## 🎯 Learning Outcomes

By completing this course, students will be able to:
- ✅ Implement transformer architectures for various biometric modalities
- ✅ Debug and optimize real-world biometric systems
- ✅ Handle extreme dataset imbalances and quality variations
- ✅ Build production-ready authentication systems
- ✅ Visualize and interpret attention mechanisms
- ✅ Apply privacy-preserving techniques to biometric data

## 📝 Assessment

- **Weekly Labs**: 40% - Hands-on implementation notebooks
- **Assignments**: 30% - Extended implementations and analysis reports
- **Final Project**: 30% - Research project on transformer-based biometrics

## 🤝 Contributing

This course is actively maintained. To report issues or suggest improvements:
- Submit GitHub issues for technical problems
- Use GitHub Discussions for course questions
- Pull requests welcome for bug fixes

## 📄 License

This course material is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to:
- Dataset providers (SOCOFing, CelebA, ASVspoof, PhysioNet)
- Open-source contributors to PyTorch, Transformers, and biometric libraries
- Course contributors and reviewers

---

**Course Repository**: [https://github.com/clarkson-edge/ee622](https://github.com/clarkson-edge/ee622)
