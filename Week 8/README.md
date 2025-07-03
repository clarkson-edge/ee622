# Week 8: Audio Transformers for Speaker Verification and Anti-Spoofing

[![Week 8 Colab](https://img.shields.io/badge/Week%208-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%208/lab/Week8_AudioTransformers.ipynb)

[← Back to Course Main](../README.md) | [← Week 7](../Week%207/README.md) | [→ Week 9](../Week%209/README.md)

## Overview
This week explores how transformer architectures revolutionize speaker verification and voice biometrics with integrated anti-spoofing capabilities. Students will implement dual-task attention-based models that simultaneously capture speaker-discriminative features and detect synthetic speech, build production-ready verification systems with liveness detection, and visualize how transformers process audio for secure biometric authentication. The comprehensive notebook guides you from audio preprocessing through deployment of a complete speaker verification system with anti-spoofing protection.

## Learning Objectives
By the end of this week, students will be able to:
- Implement transformer architectures for sequential audio processing with dual objectives
- Extract and analyze speaker embeddings while detecting spoofing attempts
- Build end-to-end speaker verification systems with enrollment, authentication, and liveness detection
- Visualize attention patterns to understand both speaker-discriminative and spoofing-indicative features
- Evaluate performance using industry-standard metrics (EER, DET curves) for both tasks
- Deploy production-ready voice biometric systems with integrated anti-spoofing protection

## 🚀 Quick Start

### Google Colab (Recommended)
Click the badge above to open the comprehensive notebook that includes:
- Complete audio preprocessing pipeline with ASVspoof 2021 dataset integration
- Dual-task transformer architecture for speaker verification and anti-spoofing
- Multi-objective training with contrastive and spoofing detection losses
- Attention mechanism visualization across layers and heads
- Interactive speaker verification demonstration with liveness detection
- Production deployment with enrollment, verification, and spoofing prevention

### Local Setup
```bash
# Use existing environment or create new
conda activate biometric-transformers

# Additional packages for Week 8
pip install torch torchaudio  # Audio processing
pip install librosa soundfile  # Advanced audio features
pip install scikit-learn  # Evaluation metrics
pip install einops  # Tensor operations
pip install matplotlib seaborn  # Visualizations
pip install kagglehub  # Dataset access
```

## 📚 Course Materials

### Comprehensive Implementation
**[Week8_AudioTransformers.ipynb](./lab/Week8_AudioTransformers.ipynb)** - Complete Audio Transformer System with Anti-Spoofing

The notebook is structured for progressive learning:

#### Part 1: Audio Processing Foundation
1. **Environment Setup** - GPU configuration and library imports
2. **ASVspoof 2021 Dataset** - Integration with logical access (LA) attacks
3. **Audio Preprocessing** - Mel-spectrograms, augmentation, normalization
4. **Preprocessing Visualization** - Waveforms, spectrograms, augmentations

#### Part 2: Dual-Task Transformer Architecture
5. **Audio Transformer Implementation** - Self-attention for sequential audio
6. **Anti-Spoofing Head** - Binary classification for liveness detection
7. **Speaker Embedding Extraction** - Attention pooling and normalization
8. **Architecture Visualization** - Model structure (3.3M parameters)

#### Part 3: Multi-Objective Training
9. **Contrastive Learning** - Speaker embedding discrimination
10. **Anti-Spoofing Loss** - Binary cross-entropy for fake detection
11. **Training Pipeline** - Simultaneous optimization of both objectives
12. **Performance Analysis** - Dual-task evaluation metrics

#### Part 4: Production System
13. **Speaker Enrollment** - Multi-sample registration with quality checks
14. **Verification Demo** - Interactive authentication with spoofing detection
15. **Robustness Analysis** - Performance under various perturbations
16. **Production Deployment** - Complete API with security features

## Key Topics Covered

### 🎓 Theory

#### Audio Transformers for Dual Tasks
- **Sequential Processing**: Treating spectrograms as frame sequences
- **Multi-Task Learning**: Shared representations for verification and anti-spoofing
- **Attention Mechanisms**: Capturing both speaker identity and synthesis artifacts
- **Position Encoding**: Preserving temporal order in utterances

#### Anti-Spoofing Features
- **Synthesis Artifacts**: Detecting unnatural patterns in TTS/VC
- **Temporal Inconsistencies**: Attention to discontinuities
- **Spectral Anomalies**: Frequency domain indicators of spoofing
- **Attack Types**: A07-A19 in ASVspoof 2021 LA scenario

#### Verification with Security
- **Dual Authentication**: Speaker match AND liveness verification
- **Embedding Quality**: Bonafide-only enrollment for security
- **Threshold Strategies**: Separate thresholds for each task
- **Score Fusion**: Combining speaker and spoofing scores

### 🛠️ Implementation Highlights

#### Audio Preprocessing Pipeline
```python
class AudioPreprocessor:
    """Production-ready audio preprocessing for ASVspoof"""
    def __init__(self, sample_rate=16000, n_mels=80, duration=3.0):
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            mel_scale="htk"
        )
        
    def load_and_preprocess(self, audio_path, augment=False):
        # Handle FLAC format from ASVspoof
        waveform, sr = torchaudio.load(audio_path)
        mel_spec = self.mel_spectrogram(waveform)
        log_mel = torch.log(mel_spec + 1e-9)
        return self.normalize(log_mel)
```

#### Dual-Task Transformer Architecture
```python
class SpeakerVerificationTransformer(nn.Module):
    """Transformer for speaker embeddings and anti-spoofing"""
    def __init__(self, input_dim=80, d_model=256, n_heads=8,
                 num_speakers=67, include_spoofing_detection=True):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff=1024)
            for _ in range(4)
        ])
        self.pooling_attention = AttentionPooling(d_model)
        
        # Dual outputs
        self.embedding_projection = nn.Linear(d_model, 256)
        self.classifier = nn.Linear(256, num_speakers)
        self.spoofing_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # Binary: bonafide vs spoofed
        )
```

#### Multi-Objective Training
```python
def train_epoch(self, epoch):
    """Train with multiple objectives"""
    # Forward pass
    outputs = self.model(mel_specs)
    
    # Three losses
    cls_loss = self.classification_loss(outputs['logits'], speaker_idxs)
    cont_loss = self.contrastive_loss(outputs['embedding'], speaker_idxs)
    spoof_loss = self.spoofing_loss(outputs['spoofing_logits'], is_bonafide)
    
    # Combined optimization
    total_loss = cls_loss + cont_loss + spoof_loss
```

## 📊 Results and Analysis

### Model Performance (10 epochs training)
| Metric | Initial | Final | Task |
|--------|---------|-------|------|
| Speaker EER | 37.7% | 15.7% | Verification |
| Speaker Accuracy | 6.8% | 57.4% | Classification |
| Spoofing EER | 21.3% | 15.7% | Anti-spoofing |
| Spoofing Accuracy | 77.9% | 82.1% | Anti-spoofing |

### ASVspoof 2021 Dataset Statistics
- **Total Samples**: 4,255 (balanced subset)
- **Bonafide**: 58.3%
- **Spoofed**: 41.7%
- **Attack Systems**: A07-A19 (neural vocoders, voice conversion)
- **Speakers**: 67 unique

### Attention Analysis
- **Untrained Model**: Uniform attention, high entropy
- **Trained Model**: Structured patterns, focused on informative regions
- **Spoofing Detection**: Attention to temporal inconsistencies
- **Speaker Verification**: Focus on stable speech regions

### Robustness Testing
| Perturbation | Mean Similarity | Std Dev | Impact |
|--------------|----------------|---------|---------|
| Clean | 0.153 | 0.277 | Baseline |
| Gaussian Noise | 0.157 | 0.277 | Minimal |
| Time Stretch | 0.144 | 0.289 | -5.9% |
| Reverb | 0.157 | 0.287 | Minimal |
| Combined | 0.156 | 0.291 | -2.0% |

## 🔬 Advanced Features

### Production Speaker Verifier with Anti-Spoofing
```python
class SpeakerVerificationDemo:
    """Complete verification system with spoofing detection"""
    def verify_speakers(self, audio_path1, audio_path2):
        # Extract embeddings and spoofing scores
        outputs1 = self.model(mel_spec1)
        outputs2 = self.model(mel_spec2)
        
        # Speaker similarity
        similarity = F.cosine_similarity(
            outputs1['embedding'], 
            outputs2['embedding']
        )
        
        # Spoofing detection
        spoof_prob1 = F.softmax(outputs1['spoofing_logits'])[1]
        spoof_prob2 = F.softmax(outputs2['spoofing_logits'])[1]
        
        # Dual decision
        same_speaker = similarity > threshold
        both_bonafide = spoof_prob1 > 0.5 and spoof_prob2 > 0.5
        
        return {
            'verified': same_speaker and both_bonafide,
            'similarity': similarity,
            'bonafide_probs': [spoof_prob1, spoof_prob2]
        }
```

### Interactive Verification Demo Features
- Real-time enrollment with spoofing checks
- Live verification with dual-task scores
- Attention visualization during verification
- Support for various attack types (TTS, voice conversion)
- Security-aware threshold adjustment

## 📝 Assignments

### Implementation Challenge (Required)
Extend the provided notebook with ONE of the following:
1. **Cross-Dataset Evaluation**: Test on VoxCeleb with synthetic attacks
2. **Advanced Anti-Spoofing**: Implement attack-specific detection heads
3. **Real-Time Processing**: Streaming verification with chunk-based attention
4. **Explainable AI**: Visualize why specific audio is flagged as spoofed

### Research Report (Required)
Write a 4-page report covering:
- Analysis of attention patterns for bonafide vs spoofed speech
- Performance comparison of single vs dual-task learning
- Evaluation across different spoofing attack types
- Recommendations for production deployment with security

**Deliverables:**
- Extended notebook with your implementation
- Technical report with experimental results
- Confusion matrices for both tasks
- Demo video showing spoofing detection in action

## 📚 Additional Resources

### 📄 Essential Papers
- [ASVspoof 2021 Challenge](https://arxiv.org/abs/2109.00535) - Dataset and baselines
- [Dual-Task Learning for Voice](https://arxiv.org/abs/2104.01292) - Multi-objective approaches
- [Attention for Anti-Spoofing](https://arxiv.org/abs/2202.05253) - Spoofing detection with transformers
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) - State-of-the-art baseline

### 🗄️ Datasets
- **[ASVspoof 2021](https://www.asvspoof.org/)** - Logical/Physical/Deepfake attacks
- **[Vox