# Video Frame Reconstruction Challenge

## Overview
This project reconstructs a jumbled video by analyzing frame similarities and determining the optimal sequential order.

## Algorithm Approach

### Method: Hybrid Similarity-Based Greedy Reconstruction

**Core Strategy:**
1. **Frame Extraction**: Extract all 300 frames from the jumbled video
2. **Similarity Matrix**: Compute pairwise similarity scores using:
   - Structural Similarity Index (SSIM) - 50% weight
   - Histogram correlation - 30% weight
   - Optical flow (motion consistency) - 20% weight
3. **Greedy Nearest Neighbor**: Build sequence by connecting most similar frames
4. **Sequence Refinement**: Optimize starting point by maximizing total path similarity

### Why This Approach?

- **SSIM**: Captures structural changes between consecutive frames
- **Histogram Correlation**: Measures color/lighting consistency
- **Optical Flow**: Estimates motion continuity between frames
- **Greedy Algorithm**: Efficiently finds good solutions in O(n²) time
- **Refinement**: Ensures we find the natural starting point of the video

### Key Design Considerations

- **Accuracy**: Multi-metric similarity ensures robust frame ordering
- **Performance**: Frame downsampling (25%) for faster computation
- **Scalability**: Works on standard laptop hardware (16GB RAM)
- **Optimization**: Vectorized operations using NumPy

### Time Complexity
- Frame extraction: O(n)
- Similarity matrix: O(n²)
- Sequence building: O(n²)
- Total: O(n²) where n = 300 frames

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download this repository**

2. **Install dependencies**:
