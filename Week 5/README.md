# Week 5: Cross-Attention Networks for Facial Attribute Analysis

[![Week 5 Colab](https://img.shields.io/badge/Week%205-Open%20in%20Colab-blue?logo=google-colab)](https://colab.research.google.com/github/clarkson-edge/ee622/blob/main/Week%205/lab/Week5_Notebook1.ipynb)

[← Back to Course Main](../README.md) | [← Week 4](../Week%204/README.md) | [→ Week 6](../Week%206/README.md)

## Overview
This week tackles one of the most challenging problems in multi-attribute learning: extreme class imbalance. Through a comprehensive journey from problem discovery to solution, students will learn how cross-attention mechanisms enable simultaneous prediction of multiple facial attributes while addressing the critical issue of models getting stuck predicting all negatives when faced with rare attributes (2% positive rate).

## Learning Objectives
By the end of this week, students will be able to:
- Understand cross-attention mechanisms and their advantages for multi-attribute tasks
- Identify and diagnose extreme dataset imbalance that kills model performance
- Implement aggressive solutions: weighted loss, focal loss, and diversity penalties
- Build interpretable cross-attention networks with visualization capabilities
- Create production-ready systems that handle rare attributes effectively
- Develop advanced visualization techniques for attention analysis

## 🚀 Quick Start

### Google Colab (Recommended)
Click the badge above to open the comprehensive notebook that includes:
- Complete environment setup with interactive visualizations
- Dataset imbalance discovery and analysis
- Failed naive training demonstration
- Aggressive balanced training implementation
- Advanced attention visualization tools
- Production-ready solutions

### Local Setup
```bash
# Use existing environment or create new
conda activate biometric-transformers

# Additional packages for Week 5
pip install einops  # Elegant tensor operations
pip install plotly  # Interactive visualizations
pip install ipywidgets  # Interactive widgets
pip install imageio  # GIF creation
pip install kagglehub  # CelebA dataset access
```

## 📚 Course Materials

### Comprehensive Implementation
**[Week5_Notebook1.ipynb](./lab/Week5_Notebook1.ipynb)** - The Complete Cross-Attention Journey

The notebook is structured as a learning journey with two main phases:

#### Phase 1: Problem Discovery and Solution
1. **Environment Setup** - Complete setup for interactive learning
2. **Understanding Cross-Attention** - Visual and conceptual introduction
3. **Architecture Implementation** - Visualizable cross-attention modules
4. **Dataset Reality Check** - Discovering extreme imbalance in CelebA
5. **Naive Training Failure** - Watch the model get stuck
6. **Aggressive Solutions** - Focal loss, 50x weights, diversity penalties
7. **Success Demonstration** - Breaking the all-negative trap

#### Phase 2: Advanced Visualization and Analysis
8. **Interactive Attention Explorer** - Imbalance-aware visualization
9. **Attention Flow Analysis** - Rare vs common attribute patterns
10. **Performance Dashboard** - Comprehensive evaluation
11. **Journey Summary** - Key takeaways and lessons learned

## Key Topics Covered

### 🎓 Theory

#### Cross-Attention Mechanism
- **Query-Key-Value Framework**: Attributes query image features
- **Multi-Head Design**: Parallel attention for different aspects
- **Learnable Embeddings**: Semantic attribute representations
- **Cross-Modal Fusion**: Bridging attributes and visual features

#### The Extreme Imbalance Problem
- **Dataset Statistics**: Some attributes at 2% positive rate
- **All-Negative Trap**: Model achieves 68.7% accuracy doing nothing
- **Standard Methods Fail**: Regular weighted loss insufficient
- **Cascading Failure**: Rare attributes completely ignored

### 🛠️ Implementation Highlights

#### Visualizable Cross-Attention Module
```python
class VisualizableCrossAttention(nn.Module):
    """Cross-attention with built-in visualization"""
    def __init__(self, query_dim, key_dim, value_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (key_dim // heads) ** -0.5
        
        # Projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        
        # Store for visualization
        self.last_attn_weights = None
```

#### Focal Loss Implementation
```python
class FocalLoss(nn.Module):
    """For extreme class imbalance"""
    def __init__(self, gamma=2, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', 
            pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()
```

#### Aggressive Weight Calculation
```python
def calculate_aggressive_pos_weights(frequencies):
    """Up to 50x weights for rare classes"""
    pos_weights = 1.0 / (frequencies + 0.001)
    pos_weights = np.clip(pos_weights, 1.0, 50.0)
    
    # Critical classes get maximum boost
    for i, freq in enumerate(frequencies):
        if freq < 0.05:  # Less than 5%
            print(f"CRITICAL: {attr_names[i]} at {freq:.1%}")
    
    return torch.tensor(pos_weights)
```

## 📊 Results and Analysis

### The Imbalance Discovery
| Attribute | Frequency | All-Negative Accuracy | Weight Applied |
|-----------|-----------|----------------------|----------------|
| Bald | 2.3% | 97.7% | 43.5x |
| Mustache | 4.1% | 95.9% | 24.4x |
| Wearing_Necktie | 5.1% | 94.9% | 19.6x |
| Young | 77.0% | 23.0% | 1.3x |

### Training Evolution
| Method | Positive Prediction Rate | F1 Score | Status |
|--------|-------------------------|----------|---------|
| Naive BCELoss | 0.0% | 0.000 | Stuck at all-negative |
| Standard Weights | 5.0% | 0.15 | Still mostly negative |
| Focal Loss (γ=2) | 18.0% | 0.42 | Breaking free |
| Aggressive Focal | 25.0% | 0.58 | Success! |

### Key Findings
1. **Standard loss completely fails** on 2% positive rate
2. **50x weights necessary** for extreme cases (not 10x)
3. **Focal loss critical** for focusing on hard examples
4. **Diversity penalties** prevent collapse during training
5. **Rare attributes need focused attention** patterns

## 🔬 Advanced Visualizations

### Interactive Attention Explorer
```python
class ImbalanceAwareAttentionExplorer:
    """Explore attention patterns with frequency context"""
    def create_interactive_viewer(self):
        # Widgets for exploration
        sample_slider = widgets.IntSlider(min=0, max=len(dataset)-1)
        layer_slider = widgets.IntSlider(min=0, max=num_layers-1)
        rarity_filter = widgets.RadioButtons(
            options=['All', 'Rare (<10%)', 'Common (>30%)']
        )
        
        # Real-time visualization updates
        def update(sample, layer, rarity):
            visualize_attention_with_context(sample, layer, rarity)
```

### Performance Dashboard Features
- ROC curves colored by attribute frequency
- F1 score vs frequency scatter plot
- Weight effectiveness visualization
- Accuracy improvement over baseline
- Per-attribute positive prediction rates

## 📝 Assignments

### Implementation Challenge (Required)
Extend the provided notebook with ONE of the following:
1. **Alternative Loss Functions**: Implement LDAM or CB loss
2. **Attention Regularization**: Add diversity loss to attention heads
3. **Curriculum Learning**: Start with balanced subset, add rare gradually
4. **Synthetic Balancing**: Generate synthetic rare attribute examples

### Analysis Report (Required)
Write a 4-page technical report covering:
- Root cause analysis of the imbalance problem
- Comparison of different loss functions and weights
- Attention pattern analysis for rare vs common attributes
- Recommendations for production deployment

**Deliverables:**
- Extended notebook with your implementation
- Technical report with experimental results
- Visualization comparing your approach to baseline
- Performance metrics table

## 📚 Additional Resources

### 📄 Essential Papers
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) - Original focal loss
- [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) - CB loss
- [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413) - LDAM
- [Cross-Attention in Transformer Architecture](https://arxiv.org/abs/2103.14899) - Cross-attention survey

### 🗄️ Datasets
- **[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** - 40 facial attributes with extreme imbalance
- **Attribute Distribution**: From 2.3% (Bald) to 77% (Young)
- **Dataset Size**: 200K images, perfect for imbalance studies

### 🛠️ Tools and Libraries
- [imbalanced-learn](https://imbalanced-learn.org/) - Sampling strategies
- [focal-loss](https://github.com/classy-vision/classy_vision/blob/master/classy_vision/losses/focal_loss.py) - Reference implementation
- [plotly](https://plotly.com/python/) - Interactive visualizations
- [einops](https://github.com/arogozhnikov/einops) - Tensor operations

## 🚨 Critical Success Factors

### Debugging Stuck Models
```python
# Quick diagnostic checklist:
1. Check positive prediction rate every epoch
2. If < 5% after 3 epochs → model is stuck
3. Double learning rate immediately
4. Increase weight cap to 100x if needed
5. Add stronger diversity penalty
```

### Red Flags to Watch For
- ❌ Accuracy looks good but all predictions negative
- ❌ F1 score near 0 for any attribute
- ❌ Loss decreasing but positive rate static
- ❌ Validation metrics worse than all-negative baseline

### Emergency Fixes
```python
# If model predicts all negatives:
if positive_rate < 0.05:
    # 1. Boost rare class weights
    pos_weights *= 2
    
    # 2. Add diversity loss
    diversity_loss = -torch.log(positive_rate + 1e-8)
    
    # 3. Increase focal loss gamma
    criterion = FocalLoss(gamma=3)  # More focus on hard examples
```

## 🎯 Skills Mastery Checklist

### Technical Skills
- ✅ Implement cross-attention from scratch
- ✅ Diagnose dataset imbalance issues
- ✅ Apply focal loss effectively
- ✅ Calculate adaptive class weights
- ✅ Build interactive visualizations

### Problem-Solving Skills
- ✅ Identify when models are "stuck"
- ✅ Debug training failures systematically
- ✅ Choose appropriate loss functions
- ✅ Balance multiple objectives
- ✅ Create production-ready solutions

### Next Week Preview
Week 6 will explore transformer encodings for iris texture representation, applying attention mechanisms to capture unique iris patterns for highly accurate biometric identification.

## 💡 Key Takeaways

1. **Extreme imbalance needs extreme measures** - Standard techniques fail at 2% positive rate
2. **Always monitor positive predictions** - High accuracy can hide complete failure
3. **Combine multiple techniques** - Focal loss + high weights + diversity penalties
4. **Visualization reveals patterns** - Rare attributes require focused attention
5. **Don't give up** - Extreme problems have solutions!

---

## 📚 Course Navigation
- [← Main Course Page](../README.md)
- [← Week 4: Vision Transformers](../Week%204/README.md)
- [→ Week 6: Iris Transformers](../Week%206/README.md)
- [Course Syllabus](../syllabus.md)
- [Reference Materials](../biometric_transformer_cheatsheet.md)