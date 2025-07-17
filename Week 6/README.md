# Week 6: Contactless Biometric Fusion - From Traditional to Transformer Approaches

[![Week 6 Colab](https://img.shields.io/badge/Week%206-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%206/lab/Week6_FusionModel.ipynb)

[← Back to Course Main](../README.md) | [← Week 5](../Week%205/README.md) | [→ Week 7](../Week%207/README.md)

## Overview
This week explores the fusion of contactless fingerprint and palmprint biometrics using quality assessment methods from Liu, Z., Zhu, B. & Du, Y. (2023). Students will implement both traditional quality assessment approaches (LQA_S and GQA_L) and modern transformer-based methods, build score-level fusion systems with quality weighting, and visualize how different modalities contribute to authentication decisions. The comprehensive notebook guides you from synthetic data generation through deployment of a complete multi-biometric fusion system.

## Learning Objectives
By the end of this week, students will be able to:
- Implement quality assessment algorithms for contactless biometrics
- Apply two-stage fusion strategies with quality weighting
- Build transformer-based biometric systems with learned quality assessment
- Generate synthetic biometric data using StyleGAN2 techniques
- Evaluate fusion performance using genuine/impostor distributions
- Deploy production-ready multi-biometric authentication systems

## 🚀 Quick Start

### Google Colab (Recommended)
Click the badge above to open the comprehensive notebook that includes:
- Complete synthetic data generation with StyleGAN2 and contactless effects
- Traditional quality assessment (LQA_S and GQA_L) implementation
- Two-stage fusion strategy from the paper
- Transformer-based quality assessment and fusion
- Interactive visualization of fusion process
- Production deployment with genuine/impostor evaluation

### Local Setup
```bash
# Use existing environment or create new
conda activate biometric-transformers

# Additional packages for Week 6
pip install torch torchvision  # Deep learning
pip install opencv-python scikit-image  # Image processing
pip install scipy scikit-learn  # Scientific computing
pip install matplotlib seaborn  # Visualizations

# For StyleGAN2 support (optional)
pip install https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl
pip install legacy
```

## 📚 Course Materials

### Comprehensive Implementation
**[Week6_FusionModel.ipynb](./lab/Week6_FusionModel.ipynb)** - Complete Biometric Fusion System

The notebook is structured for progressive learning:

#### Part 1: Environment Setup
1. **Library Imports** - PyTorch, OpenCV, StyleGAN2 dependencies
2. **Device Configuration** - GPU/CPU detection and setup

#### Part 2: Synthetic Data Generation
3. **StyleGAN2 Integration** - Advanced synthetic biometric generation
4. **ContactlessEffectsSimulator** - Perspective, blur, illumination effects
5. **QualityController** - Controllable quality degradation
6. **Dataset Generation** - 50 persons, 6 samples each

#### Part 3: Preprocessing
7. **BiometricPreprocessor** - ROI extraction and enhancement
8. **CLAHE Enhancement** - Adaptive histogram equalization
9. **Visualization** - Before/after preprocessing comparison

#### Part 4: Traditional Quality Assessment
10. **LQA_S Implementation** - SURF-based local quality
11. **GQA_L Implementation** - FFT-based global quality
12. **Quality Normalization** - Threshold-based weighting

#### Part 5: Fusion Strategy
13. **First Fusion** - Quality-weighted fingerprint combination
14. **Combined Quality** - Overall quality computation
15. **Second Fusion** - Fingerprint-palmprint combination
16. **Fusion Visualization** - Complete process demonstration

#### Part 6: Transformer Implementation
17. **QualityAssessmentTransformer** - Vision Transformer for quality
18. **BiometricTransformer** - Complete transformer system
19. **Architecture Analysis** - Model structure and parameters

#### Part 7-10: Training & Evaluation
20. **Dataset Creation** - Genuine/impostor pairs
21. **Training Pipeline** - Multi-objective optimization
22. **Performance Metrics** - EER, FAR, GAR analysis
23. **Comparison** - Traditional vs Transformer approaches

## Key Topics Covered

### 🎓 Theory

#### Quality Assessment Methods
- **LQA_S (Local Quality Assessment)**
  - SURF feature detection in original and blurred images
  - Formula: `Q_L = 1 - (n2/n1)`
  - Captures local ridge/line clarity

- **GQA_L (Global Quality Assessment)**
  - 2D-FFT spectrum analysis
  - Intensity-based quality scoring
  - Captures overall image clarity

#### Two-Stage Fusion Strategy
- **Stage 1**: Fingerprint fusion with quality weights
  - `S_f = [Σ(Q_fi^L × S_i) + Σ(Q_fi^G × S_i)] / 2`
- **Stage 2**: Fingerprint-palmprint fusion
  - `S_f_p = (Q_f/(Q_f + Q_p)) × S_f + (Q_p/(Q_f + Q_p)) × S_p`

#### Contactless Biometric Challenges
- **Perspective Distortion**: Geometric variations from capture angle
- **Defocus Blur**: Depth-of-field limitations
- **Illumination Variation**: Non-uniform lighting
- **Distance Effects**: Signal degradation with distance

### 🛠️ Implementation Highlights

#### Contactless Effects Simulation
```python
class ContactlessEffectsSimulator:
    """Simulates contactless capture conditions"""
    def simulate_contactless_capture(self, image, capture_quality='medium'):
        # Apply perspective distortion
        if distance > 10:
            result = self._apply_perspective_distortion(result, distance)
        # Apply defocus blur
        result = self._apply_defocus_blur(result, distance)
        # Apply illumination variation
        result = self._apply_illumination_variation(result, distance)
        # Apply sensor noise
        result = self._apply_sensor_noise(result, distance)
        return result
```

#### Quality Assessment Implementation
```python
class SURF_Quality_Assessment:
    """LQA_S implementation from the paper"""
    def compute_local_quality(self, image):
        # Detect features in original
        keypoints1, _ = self.feature_detector.detectAndCompute(O, None)
        # Detect features in blurred
        keypoints2, _ = self.feature_detector.detectAndCompute(O_prime, None)
        # Calculate quality
        Q_L = 1 - (n2 / n1) if n1 > 0 else 0
        return Q_L, n1, n2
```

#### Transformer-Based Quality Assessment
```python
class QualityAssessmentTransformer(nn.Module):
    """Transformer-based quality assessment"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads=8)
            for _ in range(6)
        ])
        self.local_quality_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
```

## 📊 Results and Analysis

### Model Performance (5 epochs training)
| Metric | Value | Description |
|--------|--------|-------------|
| AUC-ROC | 0.85-0.95 | Area under ROC curve |
| EER | 10-15% | Equal Error Rate |
| GAR@FAR=0.1% | 70-85% | Genuine Accept Rate |
| Training Accuracy | 78-85% | Final accuracy |

### Synthetic Dataset Statistics
- **Total Persons**: 50
- **Samples per Person**: 6
- **Total Samples**: 300
- **Biometrics**: 4 fingerprints + 1 palmprint per sample
- **Quality Range**: 0.4-1.0
- **Image Size**: 224×224 pixels

### Quality Assessment Performance
- **SURF/ORB Features**: ~500 keypoints detected
- **FFT Intensity**: 9-12 (target range)
- **Quality Correlation**: Detected quality correlates with synthetic factors

### Fusion Strategy Analysis
- **Stage 1 Weight Distribution**: Balanced across 4 fingers
- **Stage 2 Weight Ratio**: ~60% fingerprint, ~40% palmprint
- **Quality Impact**: Low quality samples receive lower fusion weights

## 🔬 Advanced Features

### StyleGAN2 Biometric Generation
```python
class StyleGAN2BiometricGenerator:
    """Advanced synthetic biometric generator"""
    def generate_fingerprint_stylegan(self, person_id, finger_id, quality_factor):
        # Generate identity and appearance codes
        identity_code = self._get_identity_code(person_id, finger_id)
        appearance_code = self._get_appearance_code(quality_factor)

        # FPGAN-Control disentanglement
        w_combined = self._disentangle_codes(w_identity, w_appearance)

        # Generate with quality degradation
        fingerprint = self.quality_controller.apply_degradation(
            fingerprint, quality_factor, unique_id
        )
        return fingerprint
```

### Two-Stage Fusion Process
```python
class PaperFusionStrategy:
    """Implementation of paper's fusion strategy"""
    def second_fusion(self, fp_score, pp_score, fp_quality, pp_quality):
        # Avoid division by zero
        total_quality = fp_quality + pp_quality
        if total_quality == 0:
            return 0.5 * (fp_score + pp_score)

        # Apply Equation 8
        fp_weight = fp_quality / total_quality
        pp_weight = pp_quality / total_quality
        S_f_p = fp_weight * fp_score + pp_weight * pp_score
        return S_f_p
```

### Traditional vs Transformer Comparison
| Aspect | Traditional (Paper) | Transformer |
|--------|-------------------|------------|
| Local Quality | SURF Features | Learned Patch Features |
| Global Quality | FFT Analysis | CLS Token Features |
| Feature Extraction | Hand-crafted | End-to-end Learned |
| Adaptability | Fixed Pipeline | Trainable |
| Computation | CPU Friendly | GPU Optimized |
| Interpretability | High | Lower |

## 📝 Assignments

### Implementation Challenge (Required)
Extend the provided notebook with ONE of the following:
1. **Cross-Modal Attention**: Add attention between fingerprint and palmprint features
2. **Quality Prediction**: Predict quality without reference (no-reference quality assessment)
3. **Adversarial Robustness**: Test fusion against presentation attacks
4. **Mobile Optimization**: Adapt for real-time mobile deployment

### Research Report (Required)
Write a 4-page report covering:
- Analysis of quality assessment effectiveness
- Comparison of fusion strategies (sum, product, quality-weighted)
- Impact of contactless effects on recognition
- Recommendations for real-world deployment

**Deliverables:**
- Extended notebook with your implementation
- Technical report with experimental results
- ROC/DET curves for different fusion methods
- Demo video showing fusion process

## 📚 Additional Resources

### 📄 Essential Papers
- [Liu et al. (2023)](https://doi.org/10.1117/12.2665422) - Original fusion paper
- [FPGAN-Control](https://arxiv.org/abs/2101.00891) - Fingerprint generation
- [QC-StyleGAN](https://arxiv.org/abs/2104.00925) - Quality control
- [Vision Transformers](https://arxiv.org/abs/2010.11929) - ViT architecture

### 🗄️ Datasets
- **[PolyU Contactless](http://www4.comp.polyu.edu.hk/)** - Contactless fingerprint
- **[CASIA Palmprint](http://biometrics.idealtest.org/)** - Palmprint database
- **[NIST SD14](https://www.nist.gov/srd/nist-special-database-14)** - Fingerprint mated pairs
- **[IIT Delhi](http://www4.comp.polyu.edu.hk/~csajaykr/IITD/)** - Touchless palmprint

### 🛠️ Tools & Libraries
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[scikit-image](https://scikit-image.org/)** - Image processing
- **[PyWavelets](https://pywavelets.readthedocs.io/)** - Wavelet transforms
- **[timm](https://github.com/rwightman/pytorch-image-models)** - PyTorch image models

### 💡 Extension Ideas
- **Multi-Spectral Fusion**: Combine visible and NIR images
- **3D Biometrics**: Extend to 3D fingerprint/palmprint
- **Continuous Authentication**: Sliding window fusion
- **Privacy-Preserving**: Homomorphic encryption for templates

## ⚡ Performance Tips

### Training Optimization
- Use mixed precision training for memory efficiency
- Implement balanced sampling for genuine/impostor pairs
- Apply progressive quality degradation during training

### Inference Optimization
- Batch quality assessment for multiple fingers
- Cache FFT computations for similar image sizes
- Use TorchScript for production deployment

### Data Efficiency
- Augment with realistic contactless effects
- Use semi-supervised learning for quality assessment
- Transfer learning from pre-trained vision models

## 🎯 Learning Outcomes

After completing this week, you will have:
- ✅ Mastered quality assessment for contactless biometrics
- ✅ Implemented multi-stage fusion strategies
- ✅ Built transformer-based biometric systems
- ✅ Generated realistic synthetic biometric data
- ✅ Evaluated fusion performance comprehensively
- ✅ Developed skills in multi-modal biometric systems

---

**Next Week**: [Week 7 - Gait Recognition with Spatial-Temporal Transformers](../Week%207/README.md) - Analyzing human gait patterns for identification at a distance.
