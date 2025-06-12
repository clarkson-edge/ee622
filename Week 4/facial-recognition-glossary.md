# Facial Recognition Glossary

## A

**Active Appearance Model (AAM)**  
Statistical model combining shape and texture variations for face modeling and recognition.

**Attention Mechanism**  
Component in transformers that learns relationships between different parts of an input, allowing the model to focus on relevant features.

**AUC (Area Under Curve)**  
Performance metric measuring the area under the ROC curve; higher values indicate better biometric system performance.

## B

**Biometric Template**  
Mathematical representation of biometric features extracted from raw data, stored for comparison during authentication.

**BIPA (Biometric Information Privacy Act)**  
Illinois state law regulating the collection, use, and storage of biometric identifiers.

## C

**Cancelable Biometrics**  
Biometric templates that can be revoked and replaced if compromised, using non-invertible transformations.

**CASIA-WebFace**  
Large-scale facial recognition dataset containing ~500K images of 10,575 subjects.

**CLS Token**  
Special classification token in transformers that aggregates information from all patches for final representation.

**Convolutional Neural Network (CNN)**  
Deep learning architecture using hierarchical feature extraction through convolutional layers.

## D

**DET Curve (Detection Error Tradeoff)**  
Plot showing trade-off between false accept and false reject rates on normal deviate scales.

**Demographic Parity**  
Fairness metric ensuring biometric system performance is consistent across demographic groups.

## E

**EBGM (Elastic Bunch Graph Matching)**  
Model-based face recognition using graphs of fiducial points with Gabor jet features.

**EER (Equal Error Rate)**  
Operating point where false accept rate equals false reject rate; lower values indicate better performance.

**Eigenfaces**  
Principal components of face images used in PCA-based face recognition.

## F

**Face Bunch Graph (FBG)**  
Collection of facial graphs representing variations in facial features across populations.

**Face Detection**  
Process of locating faces in images and determining their spatial extent.

**False Accept Rate (FAR)**  
Percentage of unauthorized users incorrectly granted access; Type II error.

**False Reject Rate (FRR)**  
Percentage of authorized users incorrectly denied access; Type I error.

**Feature Embedding**  
Compact vector representation of facial features in high-dimensional space.

**Fisherfaces**  
LDA-based face recognition maximizing between-class variance while minimizing within-class variance.

**FRVT (Face Recognition Vendor Test)**  
NIST evaluation program benchmarking face recognition algorithms.

## G

**Gabor Filter**  
Texture analysis filter mimicking human visual system response, used in face feature extraction.

**Gallery**  
Set of enrolled biometric templates against which probe samples are compared.

**Global Attention**  
Transformer mechanism allowing direct interaction between all input elements regardless of distance.

## H

**Haar Cascade**  
Classical object detection method using Haar-like features with AdaBoost classifier.

## I

**Identity Verification (1:1)**  
Biometric mode comparing probe against single claimed identity template.

**Identification (1:N)**  
Biometric mode comparing probe against entire gallery to determine identity.

**Inter-class Similarity**  
Resemblance between faces of different individuals (e.g., twins).

**Intra-class Variation**  
Differences in face images of same person due to pose, illumination, expression, aging.

**IPD (Inter-Pupillary Distance)**  
Pixel distance between eye centers; used as resolution quality metric.

**ISO/IEC 19795**  
International standard for biometric performance testing and reporting.

## L

**LBP (Local Binary Pattern)**  
Texture descriptor comparing pixel intensities in local neighborhoods.

**LDA (Linear Discriminant Analysis)**  
Supervised dimensionality reduction maximizing class separability.

**Liveness Detection**  
See Presentation Attack Detection.

## M

**Metric Learning**  
Training objective ensuring same-identity samples cluster while different identities separate.

**Morphable Model**  
3D face model with shape and texture parameters for pose-invariant recognition.

**MTCNN (Multi-task Cascaded CNN)**  
Deep learning face detector providing face bounding boxes and landmark points.

## P

**Patch Embedding**  
Process of converting image patches into vector representations for transformer processing.

**PCA (Principal Component Analysis)**  
Unsupervised dimensionality reduction capturing maximum variance in data.

**Periocular Region**  
Area around eyes including eyebrows, eyelids, and eye corners; highly discriminative for recognition.

**Positional Encoding**  
Information added to patch embeddings to preserve spatial relationships in transformers.

**Presentation Attack**  
Attempt to spoof biometric system using artifacts (photos, masks, videos).

**Presentation Attack Detection (PAD)**  
Methods to detect and prevent biometric spoofing attempts.

**Probe**  
Biometric sample presented for verification or identification.

**Prosopagnosia**  
Neurological condition impairing ability to recognize faces.

## Q

**Quality Assessment**  
Evaluation of biometric sample suitability for reliable matching (ISO/IEC 29794-5).

## R

**Rank-N Identification**  
Metric measuring if correct identity appears in top N matches.

**ROC Curve (Receiver Operating Characteristic)**  
Plot of true accept rate vs false accept rate across all thresholds.

## S

**Self-Attention**  
Mechanism computing relationships between all elements in a sequence.

**SIFT (Scale Invariant Feature Transform)**  
Local feature descriptor robust to scale and rotation changes.

**Similarity Score**  
Numerical measure of match quality between probe and gallery templates.

## T

**Template Aging**  
Degradation of biometric match performance over time due to subject changes.

**Threshold**  
Decision boundary for accepting/rejecting biometric matches.

**Triplet Loss**  
Training objective minimizing distance to positive samples while maximizing distance to negatives.

## V

**Viola-Jones Detector**  
Fast face detection using Haar features and cascaded classifiers.

**Vision Transformer (ViT)**  
Architecture applying transformer self-attention to image patches for visual recognition.

**ViT Backbone**  
Pre-trained vision transformer encoder extracting general visual features.

## W

**Within-class Variation**  
See Intra-class Variation.

## Acronyms Quick Reference

- **CNN**: Convolutional Neural Network
- **EER**: Equal Error Rate  
- **FAR**: False Accept Rate
- **FRR**: False Reject Rate
- **LBP**: Local Binary Pattern
- **LDA**: Linear Discriminant Analysis
- **PAD**: Presentation Attack Detection
- **PCA**: Principal Component Analysis
- **ROC**: Receiver Operating Characteristic
- **ViT**: Vision Transformer