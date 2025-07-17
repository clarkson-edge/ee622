# Week 9: ECG Transformers for Biometric Authentication

[![Week 9 Colab](https://img.shields.io/badge/Week%209-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%209/lab/Week9_ECGAuthentication.ipynb)

[← Back to Course Main](../README.md) | [← Week 8](../Week%208/README.md) | [→ Week 10](../Week%2010/README.md)

## Overview
This week explores how transformer architectures revolutionize ECG-based biometric authentication, offering continuous and spoof-resistant identity verification. Students will implement attention-based models that capture both local heartbeat morphology and global rhythm patterns, build production-ready authentication systems with real-time capabilities, and visualize how transformers process physiological signals for secure biometric identification. The comprehensive notebook guides you from ECG signal preprocessing through deployment of a complete authentication system using the PhysioNet ECG-ID database.

## Learning Objectives
By the end of this week, students will be able to:
- Implement transformer architectures for sequential heartbeat processing
- Extract and analyze ECG embeddings for person identification
- Build end-to-end authentication systems with enrollment and verification
- Visualize attention patterns to understand discriminative cardiac features
- Evaluate performance using biometric-specific metrics (EER, FAR, FRR)
- Deploy production-ready ECG authentication with real-time processing

## 🚀 Quick Start

### Google Colab (Recommended)
Click the badge above to open the comprehensive notebook that includes:
- Complete ECG preprocessing pipeline with PhysioNet ECG-ID integration
- Transformer architecture optimized for heartbeat sequences
- Contrastive learning for discriminative embeddings
- Attention mechanism visualization across heartbeats
- Interactive authentication demonstration
- Production deployment with enrollment and real-time verification

### Local Setup
```bash
# Use existing environment or create new
conda activate biometric-transformers

# Additional packages for Week 9
pip install wfdb neurokit2  # ECG processing
pip install torch torchvision  # Deep learning
pip install scipy scikit-learn  # Signal processing
pip install einops  # Tensor operations
pip install matplotlib seaborn  # Visualizations
```

## 📚 Course Materials

### Comprehensive Implementation
**[Week9_ECGAuthentication.ipynb](./lab/Week9_ECGAuthentication.ipynb)** - Complete ECG Transformer Authentication System

The notebook is structured for progressive learning:

#### Part 1: ECG Processing Foundation
1. **Environment Setup** - GPU configuration and library imports
2. **PhysioNet ECG-ID Dataset** - 310 recordings from 90 subjects
3. **Signal Preprocessing** - Baseline removal, normalization, QRS enhancement
4. **Preprocessing Visualization** - Raw vs filtered signals, R-peak detection

#### Part 2: Feature Engineering
5. **Heartbeat Detection** - R-peak detection with quality control
6. **Heartbeat Segmentation** - 600ms windows around R-peaks
7. **Sequence Creation** - 10-heartbeat sequences with augmentation
8. **Feature Visualization** - Heartbeat variability and templates

#### Part 3: Transformer Architecture
9. **Heartbeat Encoder** - CNN feature extraction per heartbeat
10. **Positional Encoding** - Preserving temporal order
11. **Multi-Head Attention** - Capturing inter-heartbeat patterns
12. **Architecture Analysis** - Model structure (893K parameters, 3.4MB)

#### Part 4: Training & Evaluation
13. **Person-Aware Splits** - Ensuring no data leakage
14. **Training Pipeline** - Cross-entropy and early stopping
15. **Biometric Metrics** - EER, FAR, FRR, DET curves
16. **t-SNE Visualization** - Embedding space analysis

#### Part 5: Production Deployment
17. **Authentication System** - Enrollment and verification modes
18. **Real-Time Processing** - Streaming ECG authentication
19. **Model Optimization** - TorchScript, quantization, ONNX
20. **Security Features** - Threshold tuning, continuous authentication

## Key Topics Covered

### 🎓 Theory

#### ECG as a Biometric
- **Uniqueness**: Individual cardiac morphology and rhythm
- **Liveness**: Inherent anti-spoofing (can't replay heartbeats)
- **Continuity**: Non-intrusive continuous authentication
- **Universality**: Everyone has a heartbeat

#### Transformer Advantages for ECG
- **Local Features**: CNN encoder captures heartbeat morphology
- **Global Context**: Self-attention captures rhythm patterns
- **Temporal Stability**: Handles recordings over time
- **Parallel Processing**: Efficient sequence processing

#### Biometric System Design
- **Enrollment**: Multi-recording template creation
- **Verification (1:1)**: Confirming claimed identity
- **Identification (1:N)**: Finding identity in database
- **Quality Control**: Signal quality assessment

### 🛠️ Implementation Highlights

#### ECG Preprocessing Pipeline
```python
class ECGPreprocessor:
    """Advanced ECG preprocessing for biometric authentication"""
    def __init__(self, fs=500, use_filtered=True):
        self.fs = fs
        self.use_filtered = use_filtered

    def preprocess_record(self, record):
        # Baseline wander removal
        signal = self.remove_baseline_wander(signal, cutoff=0.5)
        # Z-score normalization
        signal = self.normalize_signal(signal)
        # QRS enhancement
        signal = self.enhance_qrs(signal)
        return signal
```

#### Heartbeat Extraction
```python
class HeartbeatExtractor:
    """Extract individual heartbeats from ECG signals"""
    def __init__(self, fs=500, before_r=0.2, after_r=0.4):
        self.window_size = int((before_r + after_r) * fs)  # 300 samples

    def segment_heartbeats(self, signal, r_peaks):
        heartbeats = []
        for r_peak in r_peaks:
            start = r_peak - self.before_samples
            end = r_peak + self.after_samples
            if self._is_valid_heartbeat(heartbeat):
                heartbeats.append(heartbeat)
        return np.array(heartbeats)
```

#### ECG Transformer Architecture
```python
class ECGTransformer(nn.Module):
    """Transformer for ECG biometric authentication"""
    def __init__(self, num_classes=90, heartbeat_len=300, d_model=128):
        super().__init__()
        # Heartbeat encoder (CNN)
        self.heartbeat_encoder = HeartbeatEncoder(heartbeat_len, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8),
            num_layers=4
        )
        # Output heads
        self.classifier = nn.Linear(d_model, num_classes)
        self.embedder = nn.Linear(d_model, 256)  # For verification
```

### PhysioNet ECG-ID Dataset Statistics
- **Total Persons**: 90 (44 men, 46 women)
- **Age Range**: 13-75 years
- **Recordings**: 310 total (2-20 per person)
- **Duration**: 20 seconds at 500Hz
- **Signals**: Raw and filtered ECG lead I

### Processing Metrics
- **R-peaks detected**: ~24 per recording
- **Valid heartbeats**: ~22 per recording
- **Sequences created**: ~366 total
- **Average HR**: 72.3 ± 8.5 BPM

### Embedding Space Analysis
- **Intra-person distance**: 0.153 ± 0.277
- **Inter-person distance**: 0.847 ± 0.189
- **d-prime**: 2.89 (good separation)

## 🔬 Advanced Features

### Production Authentication System
```python
class ECGAuthenticationSystem:
    """Production-ready ECG authentication"""
    def enroll_user(self, user_id, ecg_records):
        all_embeddings = []
        for record in ecg_records:
            embeddings = self._extract_embeddings_from_record(record)
            all_embeddings.extend(embeddings)

        # Create template from average embedding
        template = np.mean(all_embeddings, axis=0)
        self.user_embeddings[user_id] = {
            'template': template,
            'embeddings': all_embeddings,
            'enrollment_date': pd.Timestamp.now()
        }
```

### Real-Time ECG Processing
```python
class RealTimeECGProcessor:
    """Simulate real-time ECG authentication"""
    def process_ecg_stream(self, ecg_signal):
        # Process in 2-second chunks
        for chunk in self.chunk_signal(ecg_signal):
            # Detect and segment heartbeats
            heartbeats = self.extract_heartbeats(chunk)

            # Authenticate when buffer full
            if len(self.heartbeat_buffer) >= 10:
                result = self.auth_system.authenticate(
                    self.heartbeat_buffer[-10:]
                )
                self.update_display(result)
```

### Model Optimization Results
| Optimization | Size | Speed | Platform |
|--------------|------|-------|----------|
| Original | 3.40 MB | 1.00x | PyTorch |
| TorchScript | 3.40 MB | 2.00x | C++ |
| Quantized | 0.85 MB | 1.50x | Mobile |
| ONNX | 3.40 MB | 1.80x | Cross-platform |

## 📝 Assignments

### Implementation Challenge (Required)
Extend the provided notebook with ONE of the following:
1. **Multi-Lead ECG**: Extend to 12-lead ECG with lead-wise attention
2. **Adversarial Robustness**: Test against ECG synthesis attacks
3. **Pathology Robustness**: Handle arrhythmias and abnormal ECGs
4. **Wearable Integration**: Adapt for single-lead smartwatch ECG

### Research Report (Required)
Write a 4-page report covering:
- Analysis of attention patterns across heartbeats
- Comparison with traditional ECG biometric methods
- Temporal stability analysis (recordings over 6 months)
- Security analysis and anti-spoofing considerations

**Deliverables:**
- Extended notebook with your implementation
- Technical report with experimental results
- Confusion matrices and DET curves
- Demo video of real-time authentication

## 📚 Additional Resources

### 📄 Essential Papers
- [ECG Biometrics Review](https://doi.org/10.1109/ACCESS.2019.2939850) - Comprehensive survey
- [Transformers for ECG](https://doi.org/10.1016/j.bspc.2021.102765) - ECG classification with attention
- [PhysioNet ECG-ID](https://physionet.org/content/ecgiddb/1.0.0/) - Dataset paper
- [Deep ECG Biometrics](https://doi.org/10.1109/TIFS.2018.2885134) - CNN approaches

### 🗄️ Datasets
- **[ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/)** - Primary dataset (used)
- **[PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/)** - Large diagnostic ECG dataset
- **[MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)** - Arrhythmia database
- **[CYBHi](http://cybhi.mty.itesm.mx/)** - Long-term ECG biometrics

### 🛠️ Tools & Libraries
- **[wfdb-python](https://github.com/MIT-LCP/wfdb-python)** - PhysioNet data reader
- **[NeuroKit2](https://neuropsychology.github.io/NeuroKit/)** - ECG analysis toolkit
- **[BioSPPy](https://github.com/PIA-Group/BioSPPy)** - Biosignal processing
- **[HeartPy](https://github.com/paulvangentcom/heartrate_analysis_python)** - Heart rate analysis

### 💡 Extension Ideas
- **Multimodal Fusion**: Combine ECG with PPG or face
- **Federated Learning**: Privacy-preserving ECG authentication
- **Edge Deployment**: Optimize for embedded devices
- **Clinical Integration**: Authentication in medical settings

## ⚡ Performance Tips

### Training Optimization
- Use mixed precision training for faster convergence
- Implement gradient accumulation for larger effective batch sizes
- Apply curriculum learning (easy → hard sequences)

### Inference Optimization
- Batch heartbeat encoding before attention
- Cache positional encodings
- Use key-value caching for streaming

### Data Efficiency
- Augment with time warping and amplitude scaling
- Use semi-supervised learning with unlabeled ECG
- Transfer learning from diagnostic ECG models

## 🎯 Learning Outcomes

After completing this week, you will have:
- ✅ Mastered ECG signal processing for biometrics
- ✅ Implemented production-ready transformer architectures
- ✅ Built complete authentication systems with enrollment/verification
- ✅ Understood biometric evaluation metrics and security
- ✅ Gained experience with real-time physiological signal processing
- ✅ Developed skills in medical signal analysis and privacy

---

**Next Week**: [Week 10 - Multimodal Biometric Fusion](../Week%2010/README.md) - Combining multiple biometric modalities with cross-attention mechanisms.
