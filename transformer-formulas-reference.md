# Transformer Mathematical Formulas Reference Sheet

## Core Transformer Architecture

### 1. Self-Attention Mechanism
The fundamental building block of transformers:

**Attention Function**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
Where:
- Q = Query matrix (n × d_k)
- K = Key matrix (n × d_k)
- V = Value matrix (n × d_v)
- d_k = Dimension of keys
- n = Sequence length

### 2. Multi-Head Attention
Allows the model to attend to different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
```
Where:
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```
- h = Number of attention heads
- W_i^Q, W_i^K, W_i^V = Projection matrices for head i
- W^O = Output projection matrix

### 3. Quality-Aware Attention
Enhanced attention mechanism for biometric applications:

```
Attention_quality(Q, K, V, q) = softmax(QK^T / √d_k ⊙ q)V
```
Where:
- q = Quality scores per patch/token
- ⊙ = Element-wise multiplication
- Quality scores weight attention based on region reliability

### 4. Positional Encoding
For sequence position information:

**Sinusoidal Encoding**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Where:
- pos = Position in sequence
- i = Dimension index
- d_model = Model dimension

**Adaptive Fingerprint Position Encoding**:
```
PE_radial(r, 2i) = sin(r/r_max / 10000^(2i/d_model))
PE_radial(r, 2i+1) = cos(r/r_max / 10000^(2i/d_model))

PE_angular(θ, 2i) = sin(θ/2π / 10000^(2i/d_model))
PE_angular(θ, 2i+1) = cos(θ/2π / 10000^(2i/d_model))
```
Where:
- r = Radial distance from detected core
- θ = Angular position relative to core
- r_max = Maximum radius in image

### 5. Feed-Forward Network
Applied to each position separately:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```
Or with GELU activation:
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

### 6. Layer Normalization
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```
Where:
- μ = Mean of x
- σ² = Variance of x
- γ, β = Learned parameters
- ε = Small constant for stability

## Biometric-Specific Formulas

### 1. Biometric Performance Metrics

**False Acceptance Rate (FAR)**:
```
FAR = Number of false acceptances / Number of impostor attempts
```

**False Rejection Rate (FRR)**:
```
FRR = Number of false rejections / Number of genuine attempts
```

**Equal Error Rate (EER)**:
Point where FAR = FRR

**D-prime (Discriminability)**:
```
d' = |μ_genuine - μ_impostor| / √(0.5(σ²_genuine + σ²_impostor))
```

### 2. Fingerprint Recognition

**Minutiae Representation**:
```
m = (x, y, θ, type)
```
Where:
- (x, y) = Spatial coordinates
- θ = Ridge orientation angle
- type ∈ {ending, bifurcation}

**Poincaré Index for Core Detection**:
```
PI(x,y) = (1/2π) ∮_C ∇θ·dl
```
For discrete implementation:
```
PI = Σ(k=0 to 3) Δθ_k
where Δθ_k = normalize(θ_{k+1} - θ_k) to [-π/2, π/2]
```
Core points: PI ≈ π/2, Delta points: PI ≈ -π/2

**Ridge Orientation Estimation**:
```
θ = 0.5 × arctan2(2×G_xy, G_xx - G_yy)
```
Where:
```
G_xx = Σ(∇_x I)²
G_yy = Σ(∇_y I)²
G_xy = Σ(∇_x I)(∇_y I)
```

**Ridge Orientation Coherence**:
```
C = √((G_xx - G_yy)² + 4G_xy²) / (G_xx + G_yy + ε)
```
Where C ∈ [0,1]: 0 = poor quality, 1 = excellent quality

**Gabor Filter for Enhancement**:
```
G(x, y, θ, f) = exp(-0.5[(x'²/σ²_x) + (y'²/σ²_y)]) × cos(2πfx')
```
Where:
```
x' = x cos(θ) + y sin(θ)
y' = -x sin(θ) + y cos(θ)
```
- σ_x, σ_y = Scale parameters
- f = Spatial frequency
- θ = Orientation

**Patch Quality Assessment**:
```
Quality = α × Clarity + β × RidgeStrength
```
Where:
```
Clarity = mean(√((∇_x I)² + (∇_y I)²))
RidgeStrength = std(I_patch)
```
- α, β = Weighting factors (typically α = β = 0.5)
- I_patch = Patch pixel intensities

**Fingerprint Similarity Score**:
```
S = Σ(matched minutiae) / min(n_1, n_2)
```
Where n_1, n_2 = Number of minutiae in templates

### 3. Iris Recognition

**2D Gabor Wavelet**:
```
G(x,y) = e^(-π[(x-x₀)²/α² + (y-y₀)²/β²]) × e^(-2πi[u₀(x-x₀) + v₀(y-y₀)])
```

**Hamming Distance for Iris Codes**:
```
HD = (IrisCodeA ⊕ IrisCodeB) ∩ MaskA ∩ MaskB / |MaskA ∩ MaskB|
```

**Geodesic Active Contour Evolution**:
```
ψ_t = -K(c + εκ)|∇ψ| + ∇ψ·∇K
```
Where:
- K = Edge stopping function
- c = Constant velocity
- κ = Curvature
- ε = Curvature weight

### 4. Face Recognition

**Patch Embedding for ViT**:
```
z_0 = [x_class; x_p^1E; x_p^2E; ...; x_p^NE] + E_pos
```
Where:
- x_class = Class token
- x_p^i = i-th image patch
- E = Patch embedding projection
- E_pos = Position embeddings

### 5. Voice/Audio Biometrics

**MFCC Computation**:
```
MFCC(k) = Σ(n=0 to N-1) log(S_mel(n)) × cos(πk(n+0.5)/N)
```
Where:
```
S_mel(m) = Σ(k=0 to N-1) |X(k)|² × H_m(k)
```
- X(k) = FFT coefficients
- H_m(k) = mel filter bank
- N = Number of filters

**Pre-emphasis Filter**:
```
H(z) = 1 - 0.97z^(-1)
```

### 6. Multimodal Fusion

**Score-Level Fusion (Weighted Sum)**:
```
s = Σ w_m · g_m(s_m)
```
Where:
- w_m = Weight for modality m
- g_m = Score normalization function
- s_m = Score from modality m

**Likelihood Ratio Fusion**:
```
Decide ω_1 if ∏(p(s_m|ω_1) / p(s_m|ω_0)) ≥ η
```

**Quality-Based Fusion**:
```
Decide ω_1 if ∏(p(s_m, q_m|ω_1) / p(s_m, q_m|ω_0)) ≥ η
```
Where q_m = Quality score for modality m

## Vision Transformer (ViT) Specific

### 1. Image to Patch Conversion
For an image of size H × W × C:
```
Number of patches N = (H × W) / P²
```
Where P = Patch size

### 2. Patch Embedding
```
x_p = Flatten(Patch) · W_E + b_E
```
Where W_E ∈ R^(P²·C × D)

### 3. Classification Head
```
y = LN(z_L^0) · W_head + b_head
```
Where z_L^0 is the class token output from the last layer

## Cross-Attention Formulas

### 1. Cross-Attention Mechanism
```
CrossAttention(Q_1, K_2, V_2) = softmax(Q_1K_2^T / √d_k)V_2
```
Where Q_1 comes from sequence 1, K_2 and V_2 from sequence 2

### 2. Cross-Modal Fusion
```
z_fused = α · SelfAttn(z_1) + β · CrossAttn(z_1, z_2)
```
Where α, β are learnable or fixed weights

## Training Objectives

### 1. Classification Loss (Cross-Entropy)
```
L_CE = -Σ y_i log(p_i)
```

### 2. Contrastive Loss (for Biometric Matching)
```
L_contrastive = (1-y) · ½D² + y · ½max(0, m-D)²
```
Where:
- y = 1 for genuine pairs, 0 for impostor
- D = Distance between embeddings
- m = Margin

### 3. Triplet Loss
```
L_triplet = max(0, D(a,p) - D(a,n) + m)
```
Where:
- a = Anchor
- p = Positive sample
- n = Negative sample

### 4. InfoNCE Loss (Contrastive Learning)
```
L_InfoNCE = -log(exp(sim(z_i,z_j)/τ) / Σ_k exp(sim(z_i,z_k)/τ))
```
Where:
- z_i, z_j = Positive pair embeddings
- τ = Temperature parameter

### 5. Quality-Weighted Loss
```
L_quality = Σ q_i × L_base(y_i, ŷ_i)
```
Where:
- q_i = Quality score for sample i
- L_base = Base loss function

## Distance and Similarity Metrics

### 1. Cosine Similarity
```
cos_sim(u,v) = (u·v) / (||u|| × ||v||)
```

### 2. Euclidean Distance
```
D_euclidean = √(Σ(u_i - v_i)²)
```

### 3. Hamming Distance (for binary codes)
```
D_hamming = Σ(u_i ⊕ v_i)
```

### 4. Mahalanobis Distance
```
D_mahalanobis = √((u-v)ᵀΣ⁻¹(u-v))
```

## Computational Complexity

### 1. Self-Attention Complexity
```
O(n² · d)
```
Where n = sequence length, d = dimension

### 2. Multi-Head Attention Complexity
```
O(n² · d + n · d²)
```

### 3. Feed-Forward Network Complexity
```
O(n · d · d_ff)
```
Where d_ff = feed-forward dimension (typically 4d)

### 4. Vision Transformer Complexity
```
O(N² · D + N · D²)
```
Where N = number of patches, D = embedding dimension

## Optimization Techniques

### 1. Learning Rate Schedule (Transformer Original)
```
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

### 2. Adam Optimizer Updates
```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - α · m_t / (√v_t + ε)
```

### 3. Cosine Annealing Schedule
```
η_t = η_min + (η_max - η_min) × (1 + cos(πT_cur/T_max)) / 2
```

## Special Considerations for Biometrics

### 1. Template Protection (Cancelable Biometrics)
```
T_protected = f(T_original, K_user)
```
Where f is a non-invertible transformation

### 2. Quality-Weighted Attention
```
Attention_quality(Q, K, V, q) = softmax(QK^T / √d_k · q)V
```
Where q = quality scores

### 3. Adaptive Threshold
```
τ_adaptive = μ_impostor + k · σ_impostor
```
Where k is chosen based on desired FAR

### 4. Score Normalization
```
s_norm = (s - μ_s) / σ_s
```
Or min-max normalization:
```
s_norm = (s - s_min) / (s_max - s_min)
```

### 5. Fusion Weight Optimization
```
w* = argmin Σ L(y_i, Σ w_m s_m^i)
```
Subject to: Σ w_m = 1, w_m ≥ 0

## Statistical Analysis

### 1. Confidence Intervals
```
CI = μ ± z_α/2 × (σ/√n)
```

### 2. Hypothesis Testing
```
z = (μ₁ - μ₂) / √(σ₁²/n₁ + σ₂²/n₂)
```

### 3. Effect Size (Cohen's d)
```
d = (μ₁ - μ₂) / σ_pooled
```
Where:
```
σ_pooled = √((σ₁² + σ₂²) / 2)
```
