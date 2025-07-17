# Contactless Biometric Fusion Glossary

## Biometric Terms

### Fingerprint
**Definition**: Unique pattern of ridges and valleys on fingertips used for identification.
**In Context**: Four fingerprints (index, middle, ring, little) are used per person.

### Palmprint
**Definition**: Pattern of lines and creases on the palm surface used for biometric identification.
**In Context**: Principal lines include heart line, head line, and life line.

### Contactless Biometric
**Definition**: Biometric capture without physical contact with sensor.
**Characteristics**: Subject to perspective distortion, defocus blur, and illumination variations.

### ROI (Region of Interest)
**Definition**: Specific area of biometric image containing discriminative features.
**Process**: Extracted using edge detection and contour analysis.

### Genuine Pair
**Definition**: Biometric samples from the same person.
**Label**: 1.0 in binary classification.

### Impostor Pair
**Definition**: Biometric samples from different persons.
**Label**: 0.0 in binary classification.

## Quality Assessment Methods

### LQA_S (Local Quality Assessment based on SURF)
**Definition**: Quality assessment using Speeded Up Robust Features detection.
**Formula**:
```
Q_L = 1 - (n2/n1)
```
Where:
- n1 = number of features in original image
- n2 = number of features in blurred image
- Q_L ∈ [0, 1]

### GQA_L (Global Quality Assessment based on FFT)
**Definition**: Quality assessment using frequency domain analysis.
**Process**:
1. Compute 2D-FFT of image
2. Extract center region intensity
3. Apply quality scoring:
```
if INT_avg > 12:
    Q_G = 100
elif INT_avg >= 9:
    Q_G = -11.11 * INT_avg² + 266.67 * INT_avg - 1500
else:
    Q_G = 0
```

### Quality Normalization
**Definition**: Normalizing quality scores above threshold for fusion weights.
**Formula** (Equation 2 for fingerprints):
```
Q_fi = Q_fi / Σ(Q_fi) if Q_fi >= threshold else 0
```

## Fusion Strategies

### First Fusion
**Definition**: Quality-weighted combination of multiple fingerprint scores.
**Formula** (Equation 6):
```
S_f = [Σ(Q_fi^L × S_i) + Σ(Q_fi^G × S_i)] / 2
```
Where:
- Q_fi^L = normalized local quality of finger i
- Q_fi^G = normalized global quality of finger i
- S_i = matching score of finger i

### Combined Quality
**Definition**: Overall quality measure for fingerprints.
**Formula** (Equation 7):
```
Q_f = (1/4) × [Σ(Q_fi^L) + Σ(Q_fi^G)]
```

### Second Fusion
**Definition**: Weighted combination of fingerprint and palmprint scores.
**Formula** (Equation 8):
```
S_f_p = (Q_f/(Q_f + Q_p)) × S_f + (Q_p/(Q_f + Q_p)) × S_p
```
Where:
- Q_f = combined fingerprint quality
- Q_p = combined palmprint quality
- S_f = fused fingerprint score
- S_p = palmprint matching score

## Image Processing Terms

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
**Definition**: Technique for improving local contrast in images.
**Parameters**: clipLimit=2.0, tileGridSize=(8,8)

### Bilateral Filter
**Definition**: Edge-preserving smoothing filter.
**Purpose**: Reduce noise while maintaining edge information.

### 2D-DFT/FFT
**Definition**: Two-dimensional Discrete/Fast Fourier Transform.
**Formula** (Equation 3):
```
F(u,v) = Σ Σ f(x,y) × e^(-j2π(ux/M + vy/N))
```

### Gabor Filter
**Definition**: Linear filter for edge detection and texture analysis.
**Use**: Enhance fingerprint ridge structure.

### Morphological Operations
**Definition**: Image processing operations based on shapes.
**Types**: Erosion, dilation, opening, closing.

## StyleGAN2 Terms

### QC-StyleGAN DegradBlock
**Definition**: Quality-controllable degradation module.
**Property**: DB(f, k×q) = k×DB(f, q) - linear degradation property.

### Quality Code
**Definition**: 16-dimensional vector controlling degradation types.
**Components**:
- Dimensions 0-3: Motion blur
- Dimensions 4-7: Gaussian noise
- Dimensions 8-11: Contrast reduction
- Dimensions 12-15: Brightness variation

### Latent Code
**Definition**: 512-dimensional vector in StyleGAN2 latent space.
**Types**: Z-space (random), W-space (mapped), W+-space (per-layer).

### Disentanglement
**Definition**: Separating identity and appearance factors in latent space.
**Strategy**: Identity controls early layers (0-8), appearance controls later layers (9-17).

## Transformer Architecture Terms

### Patch Embedding
**Definition**: Converting image into sequences of patches for transformer processing.
**Parameters**: patch_size=16, embed_dim=768

### Positional Encoding
**Definition**: Adding position information to patch embeddings.
**Purpose**: Preserve spatial relationships in transformer.

### CLS Token
**Definition**: Classification token prepended to patch sequence.
**Use**: Aggregates global image representation.

### Multi-Head Attention
**Definition**: Parallel attention mechanisms with different learned projections.
**Formula**:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### Vision Transformer (ViT)
**Definition**: Transformer architecture adapted for image processing.
**Components**: Patch embedding, position encoding, transformer blocks, classification head.

## Performance Metrics

### EER (Equal Error Rate)
**Definition**: Operating point where False Accept Rate equals False Reject Rate.
**Ideal**: Lower is better (0% = perfect).

### FAR (False Accept Rate)
**Definition**: Probability of incorrectly accepting an impostor.
**Formula**: FAR = FP / (FP + TN)

### FRR (False Reject Rate)
**Definition**: Probability of incorrectly rejecting a genuine user.
**Formula**: FRR = FN / (FN + TP)

### GAR (Genuine Accept Rate)
**Definition**: Probability of correctly accepting a genuine user.
**Relation**: GAR = 1 - FRR = TPR (True Positive Rate)

### AUC-ROC
**Definition**: Area Under the Receiver Operating Characteristic curve.
**Range**: [0, 1], where 1 = perfect classifier.

### DET Curve
**Definition**: Detection Error Tradeoff curve plotting FRR vs FAR.
**Scale**: Often uses logarithmic scale for better visualization.

## Contactless Effects

### Perspective Distortion
**Definition**: Geometric distortion due to capture angle.
**Parameters**: Maximum angle 0-15 degrees based on distance.

### Defocus Blur
**Definition**: Blur caused by depth-of-field limitations.
**Implementation**: Gaussian blur with kernel size proportional to distance.

### Illumination Variation
**Definition**: Non-uniform lighting in contactless capture.
**Model**: Multiple light sources with inverse square law falloff.

### Sensor Noise
**Definition**: Random variations in pixel values.
**Types**: Gaussian noise, salt-and-pepper noise.

## Mathematical Notations

### Quality Threshold
**Symbol**: τ (tau)
**Default**: 0.6 (60%)
**Use**: Minimum quality for inclusion in fusion.

### Matching Score
**Symbol**: S
**Range**: [0, 1]
**Subscripts**: f (fingerprint), p (palmprint), f_p (fused)

### Quality Score
**Symbol**: Q
**Range**: [0, 1]
**Superscripts**: L (local), G (global)

### Feature Count
**Symbol**: n
**Subscripts**: 1 (original), 2 (degraded)

### Frequency Intensity
**Symbol**: INT_avg
**Definition**: Average intensity in FFT spectrum center region
