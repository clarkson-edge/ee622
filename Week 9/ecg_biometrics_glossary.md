# ECG Authentication & Biometrics Glossary

## ECG Signal Components

### **P Wave**
- **Definition**: First deflection representing atrial depolarization
- **Duration**: 80-120 ms
- **Amplitude**: 0.05-0.25 mV
- **Biometric relevance**: Amplitude and morphology vary between individuals

### **QRS Complex**
- **Definition**: Sharp spike representing ventricular depolarization
- **Components**: Q (negative), R (positive), S (negative) waves
- **Duration**: 80-120 ms
- **Biometric relevance**: Most distinctive feature for person identification

### **T Wave**
- **Definition**: Represents ventricular repolarization
- **Duration**: 120-160 ms
- **Biometric relevance**: Shape and timing relative to QRS varies per person

### **R-R Interval**
- **Definition**: Time between consecutive R peaks
- **Formula**: `RR_interval = (R_peak[i+1] - R_peak[i]) / sampling_rate`
- **Heart Rate**: `HR (bpm) = 60 / RR_interval_seconds`
- **Biometric relevance**: Heart rate variability patterns are person-specific

### **Fiducial Points**
- **Definition**: Reference points in ECG signal (P onset, R peak, T offset, etc.)
- **Usage**: Alignment and segmentation of heartbeats

---

## Signal Processing Terms

### **Sampling Rate (fs)**
- **Definition**: Number of samples per second
- **ECG-ID Database**: 500 Hz (500 samples/second)
- **Nyquist frequency**: `f_nyquist = fs / 2 = 250 Hz`

### **Baseline Wander**
- **Definition**: Low-frequency drift in ECG signal
- **Causes**: Respiration, electrode movement
- **Removal**: High-pass filter with cutoff ~0.5 Hz

### **Butterworth Filter**
- **Definition**: Maximally flat magnitude response filter
- **Formula**: 
  ```
  H(s) = 1 / (1 + (s/ωc)^(2n))
  ```
  Where: ωc = cutoff frequency, n = filter order

### **Z-Score Normalization**
- **Formula**: 
  ```
  z = (x - μ) / σ
  ```
  Where: x = signal, μ = mean, σ = standard deviation
- **Purpose**: Standardize amplitude across recordings

### **Signal-to-Noise Ratio (SNR)**
- **Formula**: 
  ```
  SNR_dB = 10 * log10(P_signal / P_noise)
  ```
- **Typical ECG SNR**: 20-40 dB

---

## Feature Extraction

### **Heartbeat Segmentation**
- **Window**: Typically [-200ms, +400ms] around R peak
- **Samples per heartbeat**: 
  ```
  samples = (before_R + after_R) * sampling_rate
  = (0.2 + 0.4) * 500 = 300 samples
  ```

### **Sequence Creation**
- **Sequence length**: Number of consecutive heartbeats (e.g., 10)
- **Overlap**: Percentage of shared heartbeats between sequences
- **Stride calculation**: 
  ```
  stride = sequence_length * (1 - overlap)
  ```

### **Data Augmentation**
- **Noise injection**: `augmented = original + N(0, σ²)`
- **Amplitude scaling**: `augmented = original * (1 + U(-δ, δ))`
- Where: N = normal distribution, U = uniform distribution

---

## Transformer Architecture

### **Attention Mechanism**
- **Scaled Dot-Product Attention**:
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k) V
  ```
  Where: Q = queries, K = keys, V = values, d_k = key dimension

### **Multi-Head Attention**
- **Formula**:
  ```
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
  ```
- **Parameters**: h = number of heads (typically 8)

### **Positional Encoding**
- **Sinusoidal encoding**:
  ```
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```
  Where: pos = position, i = dimension index

### **Layer Normalization**
- **Formula**:
  ```
  LN(x) = γ * (x - μ) / σ + β
  ```
  Where: γ, β = learned parameters

### **Feed-Forward Network**
- **Structure**:
  ```
  FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
  ```
- **Typical dimensions**: d_model → 4*d_model → d_model

---

## Biometric Metrics

### **False Acceptance Rate (FAR)**
- **Definition**: Probability of incorrectly accepting an impostor
- **Formula**: 
  ```
  FAR = FP / (FP + TN)
  ```
- **Target**: < 0.01% for high security

### **False Rejection Rate (FRR)**
- **Definition**: Probability of incorrectly rejecting genuine user
- **Formula**: 
  ```
  FRR = FN / (FN + TP)
  ```
- **Target**: < 1% for user convenience

### **Equal Error Rate (EER)**
- **Definition**: Operating point where FAR = FRR
- **Calculation**: Intersection of FAR and FRR curves
- **Benchmark**: Good systems achieve EER < 5%

### **Genuine/Impostor Scores**
- **Genuine score**: Similarity between same person's samples
- **Impostor score**: Similarity between different persons' samples
- **Threshold selection**: Based on security requirements

### **Receiver Operating Characteristic (ROC)**
- **Axes**: FAR (x-axis) vs 1-FRR or TAR (y-axis)
- **Area Under Curve (AUC)**: Overall performance metric (ideal = 1.0)

### **Detection Error Tradeoff (DET)**
- **Axes**: FAR vs FRR on log scale
- **Purpose**: Better visualization of error rates

### **d-prime (d')**
- **Formula**: 
  ```
  d' = (μ_genuine - μ_impostor) / √(0.5 * (σ²_genuine + σ²_impostor))
  ```
- **Interpretation**: Separation between genuine and impostor distributions

---

## Authentication Modes

### **Verification (1:1)**
- **Definition**: Confirm claimed identity
- **Process**: Compare against single enrolled template
- **Complexity**: O(1)

### **Identification (1:N)**
- **Definition**: Determine identity from database
- **Process**: Compare against all enrolled templates
- **Complexity**: O(N)

### **Template**
- **Definition**: Stored biometric reference
- **ECG template**: Average embedding of enrolled heartbeats
- **Update strategy**: Adaptive vs fixed templates

### **Enrollment**
- **Definition**: Initial registration process
- **Requirements**: Multiple recordings for robustness
- **Quality checks**: Signal quality, sufficient heartbeats

---

## Performance Optimization

### **Quantization**
- **Dynamic quantization**: INT8 representation
- **Size reduction**: ~75% (FP32 → INT8)
- **Formula**: 
  ```
  q = round(x / scale) + zero_point
  ```

### **TorchScript**
- **Definition**: Intermediate representation for PyTorch models
- **Benefits**: JIT compilation, deployment without Python

### **ONNX**
- **Definition**: Open Neural Network Exchange format
- **Purpose**: Cross-platform deployment
- **Opset**: Version of supported operators

### **Inference Time**
- **Measurement**: Time from input to output
- **Target**: < 10ms for real-time
- **Optimization**: Batching, GPU acceleration

---

## Security Considerations

### **Liveness Detection**
- **Definition**: Ensure signal from living person
- **ECG advantage**: Inherent liveness (can't replay heartbeat)

### **Presentation Attack**
- **Definition**: Attempt to spoof biometric system
- **ECG robustness**: Difficult to synthesize realistic ECG

### **Template Protection**
- **Cancelable biometrics**: Revocable templates
- **Homomorphic encryption**: Compute on encrypted data

### **Continuous Authentication**
- **Definition**: Ongoing verification during use
- **ECG advantage**: Non-intrusive monitoring
- **Window size**: Typically 5-10 seconds

---

## Common Abbreviations

- **ECG/EKG**: Electrocardiogram
- **HR**: Heart Rate
- **HRV**: Heart Rate Variability
- **CNN**: Convolutional Neural Network
- **RNN**: Recurrent Neural Network
- **FFN**: Feed-Forward Network
- **BPM**: Beats Per Minute
- **mV**: Millivolts
- **Hz**: Hertz (cycles per second)
- **SNR**: Signal-to-Noise Ratio
- **PPG**: Photoplethysmography
- **EMD**: Empirical Mode Decomposition
- **DWT**: Discrete Wavelet Transform

---

## Key Papers & References

1. **Transformer Architecture**: Vaswani et al. (2017) - "Attention is All You Need"
2. **ECG Biometrics**: Biel et al. (2001) - "ECG analysis: a new approach in human identification"
3. **PhysioNet Database**: Goldberger et al. (2000) - "PhysioBank, PhysioToolkit, and PhysioNet"
4. **Deep ECG**: Hannun et al. (2019) - "Cardiologist-level arrhythmia detection"