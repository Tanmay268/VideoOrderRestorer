## Video Frame Reconstruction Challenge

## Overview
This project reconstructs a jumbled video by analyzing frame similarities and determining the optimal sequential order, producing a visually coherent output video.

## Algorithm Approach
Method: Hybrid Similarity-Based Greedy Reconstruction with Cluster Optimization and Boundary Fixes
Core Strategy:

Frame Extraction: Extract all frames from the input jumbled video.

Similarity Matrix: Compute pairwise frame similarity scores using multiple measures:

Structural Similarity Index (SSIM) - 25% weight

Color Histogram correlation - 15% weight

Edge Histogram correlation - 10% weight

Optical Flow motion consistency - 10% weight

CNN feature similarity (MobileNetV2) - 40% weight

Greedy Nearest Neighbor: Build the frame sequence starting from multiple candidate frames (multi-start) by connecting most similar consecutive frames.

Sequence Refinements: Optimize starting point and ordering using:

Sequence reversal check to choose best directional order

2-opt local search optimizations

Windowed 2-opt optimizations for fine-grained improvement

Clustering: Use KMeans on CNN features to group frames into clusters of visually and semantically related frames.

Cluster Direction Fix: Each cluster’s internal frame order is checked and flipped if necessary to maintain consistent motion direction using optical flow median angles.

Cluster Order Permutation: Exhaustively evaluate all permutations of cluster orderings based on smoothness score; select the best.

Boundary Transition Fix: Iteratively check transitions between consecutive clusters and flip clusters to reduce discontinuities in position or motion.

Final Direction Consistency: Confirm the final full sequence maintains consistent motion direction.

## Why This Approach?
Multimodal Similarity: Combining SSIM, histogram correlations, optical flow, and deep CNN features robustly captures perceptual, color, motion, and semantic relationships between frames.

Deep Features: MobileNetV2 embeddings enable semantic clustering and fine differentiation beyond pixel similarity.

Greedy Multi-Start & Local Search: Tackles the NP-hard ordering challenge by exploring multiple seeds and applying local reordering heuristics for optimized frame sequencing.

Clustering for Scalability & Local Consistency: Groups frames to reduce complexity and preserve temporal coherence inside clusters.

Boundary Fixes: Ensures seamless transitions between clusters, avoiding abrupt jumps or direction flips.

Efficiency & Parallelism: Downsampling frames and multithreading accelerates similarity calculations, enabling practical usage without GPUs.

Transparency & Extensibility: Modular structure allows easy extension with advanced global optimizers or neural temporal models.

## Key Design Considerations
Focus on perceptual and semantic accuracy while maintaining practical runtime.

Quadratic time complexity dominated by similarity matrix computations mitigated via parallelism.

Downsampled frames preserve key visual cues at reduced computational cost.

Use of pre-trained CNN balances domain-agnostic feature extraction with speed.

Logged quantitative metrics assess sequence smoothness, temporal SSIM, and optical flow consistency for objective quality verification.

Quality Metrics Computed
Sequence Smoothness Score: Average combined similarity of consecutive frames in the output sequence.

Temporal SSIM: Measures structural similarity between consecutive frames, reflecting temporal visual consistency.

Optical Flow Consistency: Variance-based metric assessing stability of motion directions between frames.

## Time Complexity
Step	Complexity <br>
Frame extraction	O(n)<br>
Similarity matrix	O(n²)<br>
Sequence initialization	O(n²)<br>
2-opt local search	O(k·n²) (k: max iterations)<br>
Total	O(n²)<br>
Where n = number of frames (~300).<br>

## Installation
Prerequisites
Python 3.8 or higher

pip package manager

## Setup Instructions
Clone or download this repository:

bash<br>
git clone https://github.com/Tanmay268/VideoOrderRestorer.git

## Install dependencies:

bash<br>
pip install opencv-python==4.8.1.78 numpy==1.24.3 scikit-image==0.21.0 scipy==1.11.3 tqdm==4.66.1 scikit-learn torch torchvision

## Usage
Place your jumbled video file named jumbled_video.mp4 in the project root directory.

## Run the reconstruction pipeline:

bash<br>
python frame_sorter.py
The output reconstructed video will be saved as reconstructed_video.mp4.

Execution time and quality metrics will be logged to execution_time.log.

## Logging and Output
execution_time.log includes:

Input/output video file information

Total extracted frames, FPS, resolution

Execution duration in seconds

Sequence smoothness score

Temporal SSIM score

Optical Flow Consistency score

## Potential Improvements
Integrate advanced global optimization heuristics (e.g., Simulated Annealing, Genetic Algorithms) for improved sequencing.

Utilize multi-frame contextual embeddings via LSTM or Transformer models to capture temporal dynamics.

Add motion-aware smoothness penalties to enforce temporal coherence.

Implement video frame interpolation as a post-processing step to enhance visual smoothness.

Accelerate feature extractions and similarity computations using GPU acceleration.

## For the tested video results obtained
Input   : jumbled_video.mp4

Output  : reconstructed_video.mp4

Frames  : 300

FPS     : 30

Resolution: (1920, 1080)

Execution Time           : 932.60 seconds

Sequence Smoothness Score: 88.09%

Temporal SSIM Score      : 96.12%

Optical Flow Consistency : 21.70%


## Contact
For questions or contributions, please contact: kaushiktanmay332@gmail.com.
