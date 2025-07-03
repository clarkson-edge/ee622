# Audio Transformer Speaker Verification - Glossary

## A

**Anti-spoofing**: Technology designed to detect and prevent attacks where someone tries to impersonate another person's voice using synthetic speech, recordings, or voice conversion.

**ASVspoof**: A series of challenge datasets and evaluations for developing anti-spoofing systems for automatic speaker verification. The 2021 version includes logical access (LA), physical access (PA), and deepfake (DF) scenarios.

**Attention Mechanism**: A component in transformers that allows the model to focus on different parts of the input when making predictions, learning which parts are most relevant.

**Attention Entropy**: A measure of how focused or distributed the attention weights are. High entropy means attention is spread evenly; low entropy means attention is concentrated on specific areas.

**Augmentation**: Techniques to artificially modify training data (e.g., adding noise, masking frequencies) to make models more robust and prevent overfitting.

**AUC (Area Under Curve)**: A metric that measures the area under the ROC curve, indicating overall model performance across all thresholds. Values closer to 1 indicate better performance.

## B

**Bonafide**: Genuine, authentic speech from a real human speaker, as opposed to spoofed or synthetic speech.

**Batch Size**: The number of samples processed together in one forward/backward pass during training.

## C

**Closed-set Identification**: Speaker identification where the speaker must be one of a known set of enrolled speakers.

**Contrastive Loss**: A loss function that pulls embeddings of the same speaker closer together while pushing embeddings of different speakers apart.

**Cosine Similarity**: A metric measuring the cosine of the angle between two vectors, used to compare speaker embeddings. Values range from -1 to 1, with 1 indicating identical directions.

## D

**Deepfake**: Artificially generated or manipulated audio/video content that appears authentic, often created using deep learning techniques.

**d_model**: The dimensionality of the model's internal representations in the transformer architecture.

## E

**EER (Equal Error Rate)**: The point where false acceptance rate equals false rejection rate. Lower EER indicates better system performance.

**Embedding**: A dense vector representation of a speaker's voice characteristics, extracted by the neural network.

**Enrollment**: The process of registering a new speaker in the system by extracting and storing their voice characteristics.

## F

**FAR (False Acceptance Rate)**: The percentage of unauthorized speakers incorrectly accepted by the system.

**FFT (Fast Fourier Transform)**: An algorithm to compute the frequency spectrum of a signal.

**FLAC**: Free Lossless Audio Codec, an audio format that compresses without quality loss.

**Frequency Masking**: An augmentation technique that randomly masks frequency bands in the spectrogram.

**FRR (False Rejection Rate)**: The percentage of authorized speakers incorrectly rejected by the system.

## G

**GELU (Gaussian Error Linear Unit)**: An activation function used in transformers that provides smooth, non-linear transformations.

## H

**Hop Length**: The number of samples between successive frames when computing spectrograms.

## I

**Identification**: Determining who is speaking from a set of enrolled speakers.

## L

**LA (Logical Access)**: A type of spoofing attack using text-to-speech or voice conversion systems.

**Log Mel-Spectrogram**: A representation of audio that shows frequency content over time, with frequencies mapped to the mel scale and amplitudes in log scale.

**Loss Function**: A function that measures how wrong the model's predictions are, used to guide training.

## M

**Mel Scale**: A perceptual scale of pitches that approximates human hearing, where equal distances sound equally spaced to listeners.

**Mel-Spectrogram**: A spectrogram where frequencies are converted to the mel scale.

**Multi-Head Attention**: Using multiple attention mechanisms in parallel, each potentially learning different relationships in the data.

## N

**n_fft**: The number of samples used in each FFT window when computing spectrograms.

**n_heads**: The number of attention heads in multi-head attention.

**n_mels**: The number of mel frequency bins in a mel-spectrogram.

## O

**Open-set Identification**: Speaker identification where the speaker might not be in the enrolled set.

## P

**PA (Physical Access)**: A type of spoofing attack using replay of recorded speech.

**Positional Encoding**: Information added to embeddings to give the transformer information about the position of elements in a sequence.

**Presentation Attack**: An attempt to spoof a biometric system by presenting fake biometric data.

## R

**Replay Attack**: A spoofing attack where a recording of genuine speech is played back to the system.

**ROC Curve (Receiver Operating Characteristic)**: A graph showing the trade-off between true positive rate and false positive rate at various thresholds.

## S

**Sample Rate**: The number of audio samples per second, typically 16kHz for speech.

**Self-Attention**: Attention mechanism where a sequence attends to itself, learning relationships between different positions.

**Speaker Embedding**: A fixed-size vector representation that captures speaker-specific characteristics.

**Speaker Verification**: Determining whether a speech sample matches a claimed identity.

**Spectrogram**: A visual representation of the frequency spectrum of a signal over time.

**Spoofing**: Attempting to deceive a biometric system by presenting fake biometric traits.

## T

**t-SNE (t-distributed Stochastic Neighbor Embedding)**: A dimensionality reduction technique for visualizing high-dimensional data in 2D or 3D.

**Text-to-Speech (TTS)**: Technology that converts written text into synthetic speech.

**Threshold**: A decision boundary value used to determine whether to accept or reject a verification claim.

**Time Masking**: An augmentation technique that randomly masks time segments in the spectrogram.

**Transformer**: A neural network architecture based on self-attention mechanisms, originally developed for NLP but applicable to many domains.

## V

**Verification**: The process of confirming whether a speaker is who they claim to be.

**Voice Conversion**: Technology that modifies speech to sound like a different speaker while preserving content.

**Vocoder**: A system that synthesizes speech from acoustic features.

## W

**Win Length**: The length of each window used when computing the short-time Fourier transform.

**Window Function**: A function applied to each frame of audio before FFT to reduce spectral leakage.