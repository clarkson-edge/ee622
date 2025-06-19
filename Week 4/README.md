# Week 4: Vision Transformers (ViT) for Facial Recognition

[![Week 4 Colab](https://img.shields.io/badge/Week%204-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%204/lab/Week4_FaceBiometrics.ipynb)

[← Back to Course Main](../README.md) | [← Week 3](../Week%203/README.md) | [→ Week 5](../Week%205/README.md)

## Overview
This week explores how Vision Transformers (ViT) revolutionize facial recognition by treating face images as sequences of patches. Students will implement ViT architectures for face recognition, compare them with traditional CNN approaches, and visualize how transformers attend to different facial features.

## Learning Objectives
By the end of this week, students will be able to:
- Understand ViT architecture and its adaptation for face biometrics
- Implement face image patching and position encoding strategies
- Build transformer-based face recognition and verification systems
- Visualize and interpret facial feature attention patterns
- Compare ViT performance with CNN-based face recognition
- Integrate transformers with existing face recognition pipelines

## 🚀 Quick Start

### Google Colab (Recommended)
Click the badge above to open the notebook directly from GitHub. The comprehensive notebook includes:
- ViT implementation from scratch
- Face detection and alignment preprocessing
- Transformer-based feature extraction
- Face verification and identification tasks
- Attention visualization across facial regions

### Local Setup
```bash
# Use existing environment or create new
conda activate biometric-transformers

# Additional packages for Week 4
pip install face-recognition  # For face detection
pip install dlib  # Face alignment
pip install timm  # Pre-trained ViT models
```

## 📚 Course Materials

### Hands-On Implementation
**[Week4_FaceBiometrics.ipynb](./lab/Week4_FaceBiometrics.ipynb)** - Complete ViT for Face Recognition
- Face preprocessing pipeline (detection, alignment, normalization)
- ViT architecture implementation with facial adaptations
- Training on face datasets (LFW, CelebA)
- Face verification (1:1) and identification (1:N)
- Attention visualization and interpretation
- Performance comparison with FaceNet, ArcFace

### Theory Slides
**[Week4_FaceBiometrics.pptx](./slides/Week4_FaceBiometrics.pptx)** - Conceptual foundations
- Vision Transformer architecture overview
- Patch embedding strategies for faces
- Position encoding in facial context
- Multi-scale ViT approaches
- Hybrid CNN-Transformer architectures

### Course Notes
**Chapter 4 Notes** - Available in multiple formats:
- [EE622 Chapter 4.md](./notes/EE622%20Chapter%204.md) - Markdown format
- [EE622 Chapter 4.pdf](./notes/EE622%20Chapter%204.pdf) - PDF for printing
- [EE622 Chapter 4.odt](./notes/EE622%20Chapter%204.odt) - OpenDocument format

### Reference Materials
**[facial-recognition-glossary.md](./facial-recognition-glossary.md)** - Key terms and concepts
- Face recognition terminology
- ViT-specific definitions
- Performance metrics for face biometrics
- State-of-the-art comparisons

## Key Topics Covered

### 🎓 Theory

#### Vision Transformers for Faces
- **Patch-based Processing**: Dividing face images into 16x16 patches
- **Facial Position Encoding**: Encoding spatial relationships between facial parts
- **Class Token**: Aggregating facial features for recognition
- **Multi-Head Attention**: Learning relationships between facial regions

#### Advantages over CNNs
- **Global Context**: Capturing long-range facial dependencies
- **Interpretability**: Visualizing which patches matter
- **Flexibility**: Handling varying face sizes and poses
- **Transfer Learning**: Leveraging pre-trained ViT models

### 🛠️ Implementation Highlights

#### Face Preprocessing Pipeline
```python
def preprocess_face_for_vit(image_path, target_size=224):
    """Detect, align, and prepare face for ViT"""
    # Detect face
    face_locations = face_recognition.face_locations(image)
    
    # Align face using facial landmarks
    face_landmarks = face_recognition.face_landmarks(image)
    aligned_face = align_face(image, face_landmarks)
    
    # Resize and normalize
    face_tensor = transform(aligned_face)
    return face_tensor
```

#### ViT Architecture for Faces
```python
class FaceViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=12, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        
        # Face-specific head
        self.face_head = nn.Linear(embed_dim, num_identities)
```

#### Attention Visualization
```python
def visualize_face_attention(model, face_image, layer_idx=-1):
    """Visualize which facial regions the model focuses on"""
    # Get attention weights
    _, attentions = model(face_image, output_attentions=True)
    
    # Average attention across heads
    attention_map = attentions[layer_idx].mean(dim=1)
    
    # Reshape to image dimensions
    attention_img = attention_map.reshape(14, 14)
    attention_img = F.interpolate(attention_img, size=(224, 224))
    
    # Overlay on original face
    return overlay_attention(face_image, attention_img)
```

## 📊 Results and Analysis

### Performance Metrics
| Model | LFW Accuracy | Params | Inference Time |
|-------|-------------|---------|----------------|
| FaceNet (CNN) | 99.65% | 22M | 15ms |
| ArcFace (CNN) | 99.82% | 34M | 18ms |
| ViT-B/16 (Ours) | 99.73% | 86M | 25ms |
| Hybrid CNN-ViT | 99.85% | 45M | 20ms |

### Key Findings
1. **Attention Patterns**: ViT focuses on eyes, nose, and mouth regions
2. **Pose Robustness**: Better handling of profile views than CNNs
3. **Data Efficiency**: Requires more training data than CNNs
4. **Interpretability**: Clear visualization of important facial features
5. **Hybrid Benefits**: CNN backbone + ViT achieves best results

### Attention Analysis
```
Layer 1-3: Low-level features (edges, textures)
├── Attention spread across entire face
└── No specific feature focus

Layer 4-6: Mid-level features
├── Emerging focus on facial landmarks
└── Eyes and nose regions highlighted

Layer 7-12: High-level features
├── Strong attention on discriminative regions
├── Eyes: 35% of attention
├── Nose: 25% of attention
├── Mouth: 20% of attention
└── Face contour: 20% of attention
```

## 🔬 Advanced Topics

### Multi-Scale ViT
```python
class MultiScaleViT(nn.Module):
    """Process face at multiple resolutions"""
    def __init__(self):
        self.fine_vit = ViT(img_size=224, patch_size=14)
        self.coarse_vit = ViT(img_size=224, patch_size=28)
        
    def forward(self, x):
        fine_features = self.fine_vit(x)
        coarse_features = self.coarse_vit(x)
        return torch.cat([fine_features, coarse_features], dim=-1)
```

### Face-Specific Position Encoding
```python
def create_facial_position_encoding(num_patches, embed_dim):
    """Position encoding aware of facial structure"""
    # Standard 2D sinusoidal encoding
    pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5))
    
    # Add facial region bias
    facial_regions = define_facial_regions()  # Eyes, nose, mouth areas
    for region, weight in facial_regions.items():
        pos_embed[region] *= weight
        
    return pos_embed
```

## 📝 Assignments

### Implementation Tasks (Choose 2)
1. **ViT Variants**: Implement DeiT or Swin Transformer for faces
2. **Augmentation Study**: Test ViT robustness to facial occlusions
3. **Attention Analysis**: Compare attention patterns across ethnicities
4. **Speed Optimization**: Implement efficient ViT inference

### Comparative Analysis (Required)
Create a comprehensive comparison between CNN and ViT approaches:
- Architecture differences and complexity
- Training data requirements
- Inference speed and accuracy trade-offs
- Interpretability and explainability
- Practical deployment considerations

**Deliverable Format:**
- Jupyter notebook with experiments
- 4-page report with visualizations
- Performance comparison table
- Recommendation for production use

## 📚 Additional Resources

### 📄 Key Papers
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Original ViT paper
- [Face Transformer for Recognition](https://arxiv.org/abs/2103.14803) - Face-specific adaptations
- [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877) - DeiT improvements
- [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832) - CNN baseline

### 🗄️ Datasets
- **[LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)** - Standard benchmark
- **[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** - Large-scale face attributes
- **[MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)** - Million-scale training
- **[VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)** - Diverse poses and ages

### 🛠️ Pre-trained Models
- [timm (PyTorch Image Models)](https://github.com/rwightman/pytorch-image-models) - Pre-trained ViTs
- [face-recognition](https://github.com/ageitgey/face_recognition) - Face detection/alignment
- [InsightFace](https://github.com/deepinsight/insightface) - SOTA face recognition models

## 🚨 Common Issues and Solutions

### "Out of Memory with ViT"
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Use gradient accumulation
accumulation_steps = 4

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### "Poor Face Detection"
```python
# Try multiple detectors
detectors = ['dlib', 'mtcnn', 'retinaface']
for detector in detectors:
    faces = detect_faces(image, detector=detector)
    if len(faces) > 0:
        break
```

### "Attention Visualization Unclear"
```python
# Use later layers for clearer patterns
attention = model.get_attention(layer=-2)  # Second to last

# Apply threshold for clarity
attention[attention < 0.1] = 0

# Use different colormap
plt.imshow(attention, cmap='jet', alpha=0.5)
```

## 🎯 Learning Path Progress

### Prerequisites Completed
- ✅ Week 1: Transformer fundamentals
- ✅ Week 2: Attention mechanisms
- ✅ Week 3: Biometric applications

### Skills Developed This Week
- ✅ ViT architecture implementation
- ✅ Face-specific adaptations
- ✅ Attention interpretation
- ✅ Performance benchmarking
- ✅ Hybrid architectures

### Next Steps
- → Week 5: Cross-attention for facial attributes
- → Multi-modal face analysis
- → Real-time deployment optimization

## 🔄 Next Week Preview
Week 5 will explore cross-attention networks for facial attribute analysis, learning to predict multiple attributes simultaneously while handling extreme class imbalance in facial datasets.

---

## 📚 Course Navigation
- [← Main Course Page](../README.md)
- [← Week 3: Self-Attention for Minutiae](../Week%203/README.md)
- [→ Week 5: Cross-Attention Networks](../Week%205/README.md)
- [Course Syllabus](../syllabus.md)
- [Reference Materials](../biometric_transformer_cheatsheet.md)