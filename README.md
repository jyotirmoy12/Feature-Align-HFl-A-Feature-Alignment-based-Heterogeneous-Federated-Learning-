# Feature Align-HFL: Feature Alignment-Based Client-Specific Model for Heterogeneous Federated Learning

## Overview
Feature Align-HFL is a novel framework that enables federated learning across heterogeneous model architectures through feature alignment. This approach allows different organizations to collaborate and share knowledge while maintaining their specialized model architectures and preserving data privacy.

## Key Features
- Support for heterogeneous model architectures in federated learning
- Privacy-preserving feature extraction and alignment
- Dimensionality reduction using PCA
- Cosine similarity-based feature matching
- Common feature aggregation and redistribution

## Architecture
The Feature Align-HFL framework consists of several key components:
1. Local model training on private data
2. Feature extraction and dimensionality reduction
3. Feature alignment using cosine similarity
4. Identification and aggregation of common features
5. Feature redistribution to client models

## Technical Details

### Feature Alignment
The core feature alignment process uses cosine similarity to measure the directional similarity between feature vectors:

```
Sc(ft(xj), fs(xj)) = ft(xj) · fs(xj) / (||ft(xj)|| ||fs(xj)||)
```

Where:
- ft(xj) and fs(xj) are reduced feature vectors of clients t and s
- || · || denotes the L2 norm (Euclidean norm)

### Common Feature Aggregation
Common features are identified using a threshold-based approach:
```
Fcom = {f ∈ fa, fb : Sc(fa(f), fb(f)) > τ}
```

## Experimental Results

### Dataset
- CIFAR-10
- 50,000 training images
- 10,000 test images
- 10 classes
- 32x32 pixel RGB images

### Model Performance
- CNN: 92.32% accuracy, 0.22 loss (33 epochs)
- U-Net: 87.60% accuracy, 0.35 loss (33 epochs)

### Threshold Analysis
Different threshold values (τ) were tested for feature alignment:
- τ = 0.3: Higher number of common features, lower similarity requirement
- τ = 0.5: Balanced trade-off between feature count and similarity
- τ = 0.7: Fewer but more strongly correlated features

## Key Findings
1. Successfully demonstrated feature alignment between heterogeneous models (CNN and U-Net)
2. Identified meaningful shared features across different architectures
3. Threshold selection significantly impacts the quality and quantity of aligned features
4. Effective knowledge transfer while maintaining model specialization



## Contact
For questions or collaborations, contact:
- Jyotirmoy Nath - jyotirmoy.nath23m@iiitg.ac.in



## Acknowledgments
This work was presented at the IEEE 12th International Conference on Orange Technology (ICOT 2024).
