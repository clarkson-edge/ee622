# Week 2: Transformer Models for Fingerprint Feature Extraction and Matching

[![Week 2 Colab](https://img.shields.io/badge/Week%202-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%202/lab/week2_nb1_fingerprint_transformer.ipynb)

[← Back to Course Main](../README.md) | [← Week 1](../Week%201/README.md)

## Overview
This week explores the application of transformer architectures to fingerprint biometrics, focusing on hybrid CNN-transformer models that combine local feature extraction with global context understanding. Students will implement a complete fingerprint recognition pipeline using the SOCOFing dataset with advanced features like core detection and quality-aware attention.

## Learning Objectives
By the end of this week, students will be able to:
- Design hybrid CNN-transformer architectures for fingerprint feature extraction
- Implement quality-aware processing for challenging fingerprint scenarios
- Apply Poincaré index for fingerprint core detection and adaptive position encoding
- Develop core-focused attention analysis techniques
- Evaluate transformer advantages over traditional minutiae-based methods
- Work with real-world fingerprint datasets (SOCOFing)

## 🚀 Quick Start

### Google Colab (Recommended)
Click the Colab badge above to open the complete implementation directly from GitHub. The notebook includes:
- **SOCOFing Dataset Integration**: Automatic download via kagglehub
- **Advanced Preprocessing**: Gabor filtering, orientation estimation, quality assessment
- **Core Detection**: Poincaré index-based singular point detection
- **Quality-Aware Attention**: Patch-level quality scoring and attention weighting

### Local Setup
```bash
# Create conda environment
conda create -n fingerprint-transformers python=3.9
conda activate fingerprint-transformers

# Install required packages
pip install torch torchvision transformers
pip install opencv-python scikit-image matplotlib seaborn
pip install kagglehub numpy scipy pandas
```

## 📚 Course Materials

### Theory Materials
- **[Lecture Slides](./slides/week2_theory_fingerprint_transformers.pptx)** - Complete theoretical foundation

### Hands-On Implementation
- **[Main Notebook](./lab/week2_nb1_fingerprint_transformer.ipynb)** - Complete fingerprint transformer implementation with:
  - SOCOFing dataset loading and exploration
  - Advanced preprocessing pipeline
  - Hybrid CNN-transformer architecture
  - Core-focused attention analysis
  - Multi-layer attention evolution study

## Key Topics Covered

### 🎓 Theory (25-minute lecture)
1. **Fingerprint Recognition Fundamentals**
   - Three levels of fingerprint features (Level 1: patterns, Level 2: minutiae, Level 3: pores)
   - Traditional minutiae-based vs. transformer approaches
   - Challenges: quality variations, partial prints, cross-sensor matching

2. **Hybrid CNN-Transformer Architectures**
   - Design principles for fingerprint-specific models
   - Local feature extraction (CNN) + global context (transformer)
   - Quality-aware attention mechanisms

3. **Mathematical Foundations**
   - Poincaré index for core detection: `PI = Σ(k=0 to 3) Δθ_k`
   - Ridge orientation estimation: `θ = 0.5 × arctan2(2×G_xy, G_xx - G_yy)`
   - Quality assessment: `Quality = α × Clarity + β × RidgeStrength`

### 🛠️ Hands-On Implementation (20-minute demo)
1. **SOCOFing Dataset Processing**
   - Real African fingerprint data (100+ samples)
   - Automatic dataset download via kagglehub
   - Subject-specific analysis and visualization

2. **Advanced Preprocessing Pipeline**
   - **Ridge Enhancement**: Gabor filtering with 16 orientations
   - **Orientation Field Estimation**: Gradient tensor method
   - **Core Detection**: Poincaré index calculation (PI ≈ π/2 for cores)
   - **Quality Assessment**: Patch-level clarity and ridge strength scoring

3. **Transformer Architecture Features**
   - **Adaptive Position Encoding**: Relative to detected fingerprint core
   - **Quality-Aware Attention**: Weighted by patch quality scores
   - **Multi-Head Processing**: Different attention heads for different minutiae types
   - **Patch-Based Tokenization**: 32×32 patches with 50% overlap

4. **Advanced Analysis Tools**
   - **Core-Focused Attention Analysis**: Correlation between attention and distance from core
   - **Layer-Wise Evolution**: How attention patterns develop through transformer layers
   - **Regional Analysis**: Core vs. middle vs. peripheral attention distribution
   - **Quality Correlation**: Attention weights vs. patch quality scores

## 🔬 Implementation Highlights

### Core Detection Algorithm
```python
def detect_fingerprint_core(fingerprint, orientations):
    """Detect core using Poincaré index"""
    # Calculate Poincaré index for each 2x2 block
    poincare = compute_poincare_index(orientations)

    # Find cores (PI ≈ π/2)
    core_candidates = find_singularities(poincare, threshold=π/3)

    # Return strongest candidate
    return select_best_core(core_candidates, fingerprint)
```

### Quality-Aware Attention
```python
def compute_patch_quality(patches):
    """Assess patch quality for attention weighting"""
    quality_scores = []
    for patch in patches:
        clarity = np.mean(np.sqrt(gradient_magnitude(patch)))
        ridge_strength = np.std(patch)
        quality = (clarity + ridge_strength) / 2
        quality_scores.append(quality)
    return normalize_scores(quality_scores)
```

### Adaptive Position Encoding
```python
def adaptive_position_encoding(patch_positions, core_point, d_model):
    """Generate core-relative position encodings"""
    encodings = []
    for (x, y) in patch_positions:
        # Calculate radial distance and angle from core
        radius = np.sqrt((x - core_point[0])**2 + (y - core_point[1])**2)
        angle = np.arctan2(y - core_point[1], x - core_point[0])

        # Generate encoding
        encoding = generate_radial_angular_encoding(radius, angle, d_model)
        encodings.append(encoding)
    return encodings
```

## 📊 Results and Analysis

The implementation demonstrates several key findings:

### Core-Focused Attention Patterns
- **Distance Correlation**: 0.4618 correlation between attention and distance from core
- **Regional Distribution**: Peripheral regions receive higher attention than core regions
- **Quality Correlation**: Strong positive correlation between patch quality and attention weights

### Layer-Wise Evolution
- **Layer 1**: Strong structural focus (correlation: 0.100)
- **Layer 2-4**: Decreasing structural correlation, increasing semantic focus
- **Final Layers**: Balanced attention distribution for robust matching

### Performance Metrics
- **Dataset**: 100 SOCOFing fingerprint samples processed
- **Core Detection**: Successful detection in 95%+ of samples
- **Processing Speed**: Real-time capable with GPU acceleration
- **Quality Assessment**: Reliable patch scoring for attention weighting

## 📝 Assignments
1. **Implementation**: Run the complete notebook and analyze attention patterns
2. **Comparison**: Compare quality-aware vs. uniform attention performance
3. **Analysis**: Write a report on core-focused attention findings
4. **Extension**: Experiment with different patch sizes or quality metrics

**Deliverables:**
- Completed notebook with attention visualizations
- Comparison of attention evolution across transformer layers

## 📚 Additional Resources

### 📄 Key Papers
- [Transformer based Fingerprint Feature Extraction (IEEE 2022)](https://ieeexplore.ieee.org/document/9956435)
- [AFR-Net: Attention-Driven Fingerprint Recognition Network](https://arxiv.org/abs/2211.14297)
- [Deep Learning for Fingerprint Recognition: A Survey](https://ieeexplore.ieee.org/document/10158009)

### 🗄️ Datasets
- **[SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing)** - Real African fingerprint data (used in notebook)
- [FVC2004 Fingerprint Verification Competition](http://bias.csr.unibo.it/fvc2004/)
- [NIST Special Database 302](https://www.nist.gov/srd/nist-special-database-302)

### 🛠️ Tools and Libraries
- [NIST Biometric Image Software](https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis)
- [Fingerprint Enhancement Python](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python)
- [Kagglehub](https://github.com/Kaggle/kagglehub) - Dataset access

## 🔄 Next Week Preview
Week 3 will dive deeper into self-attention mechanisms specifically designed for minutiae detection and ridge analysis, exploring how transformers can replace traditional feature extraction methods entirely while maintaining compatibility with existing biometric systems.

---

## 📚 Course Navigation
- [← Main Course Page](../README.md)
- [← Week 1: Transformer Fundamentals](../Week%201/README.md)
- [Course Syllabus](../syllabus.md)
- [Reference Materials](../biometric_transformer_cheatsheet.md)

## 🆘 Support
- **Course Repository**: [https://github.com/clarkson-edge/ee622](https://github.com/clarkson-edge/ee622)
- **Issues**: Submit GitHub issues for technical problems
- **Discussions**: Use GitHub Discussions for course questions
