# Video Frame Reconstruction Challenge

## Overview
This project reconstructs a jumbled video by analyzing frame similarities and determining the optimal sequential order, producing a visually coherent output video.

## Algorithm Approach

### Method: Hybrid Similarity-Based Greedy Reconstruction

**Core Strategy:**
1. **Frame Extraction:** Extract all frames from the input jumbled video.
2. **Similarity Matrix:** Compute pairwise frame similarity scores using multiple measures:
   - Structural Similarity Index (SSIM) - 25% weight
   - Color Histogram correlation - 15% weight
   - Edge Histogram correlation - 10% weight
   - Optical Flow motion consistency - 10% weight
   - CNN feature similarity (MobileNetV2) - 40% weight
3. **Greedy Nearest Neighbor:** Build the frame sequence starting from multiple candidate frames (multi-start) by connecting most similar consecutive frames.
4. **Sequence Refinements:** Optimize starting point and ordering using:
   - Sequence reversal check to choose best directional order
   - 2-opt local search optimizations
   - Windowed 2-opt optimizations for fine-grained improvement

## Why This Approach?

- **SSIM:** Captures fine structural changes between frames, important for perceptual visual similarity.
- **Histogram Correlation:** Ensures color and lighting consistency across frames.
- **Optical Flow:** Encodes motion continuity by analyzing frame-to-frame flow magnitudes.
- **Deep CNN Features:** Provides semantic frame representations capturing complex visual content.
- **Multi-Start Greedy:** Mitigates poor local optima by exploring multiple sequence seeds.
- **Local Search:** Refines the sequence order to maximize total frame similarity cohesiveness.

## Key Design Considerations

- **Perceptual Quality Focus:** Uses multimodal similarity metrics to robustly capture visual and semantic relationships.
- **Efficiency:** Downsamples frames to 320x180 for faster processing while maintaining key visual features.
- **Scalability:** Designed to run efficiently on general hardware without GPU dependency.
- **Parallelization:** Uses multithreading to accelerate pairwise similarity computations.
- **Simplicity and Modularity:** Employs well-known algorithms suitable for NP-hard ordering problems without complex black-box models.

## Quality Metrics Computed

- **Sequence Smoothness Score:** Average combined similarity of consecutive frames in the output sequence.
- **Temporal SSIM:** Measures structural similarity between consecutive frames, reflecting temporal visual consistency.
- **Optical Flow Consistency:** Variance-based metric assessing stability of motion directions between frames.

## Time Complexity

| Step                  | Complexity |
|-----------------------|------------|
| Frame extraction      | O(n)       |
| Similarity matrix     | O(n²)      |
| Sequence initialization | O(n²)    |
| 2-opt local search    | O(k·n²) (k: max iterations) |
| Total                 | O(n²)      |

Where n = number of frames (~300).

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download this repository:
git clone <repository_url>
cd <repository_directory>


2. Install dependencies:
pip install opencv-python numpy scikit-image torch torchvision scipy


## Usage

1. Place your jumbled video file named `jumbled_video.mp4` in the project root directory.

2. Run the reconstruction pipeline:
python frame_sorter.py


3. The output reconstructed video will be saved as `reconstructed_video.mp4`.

4. Execution time and quality metrics will be logged to `execution_time.log`.

## Logging and Output

- `execution_time.log` includes:
  - Input/output video file information
  - Total extracted frames, FPS, resolution
  - Execution duration in seconds
  - Sequence smoothness score
  - Temporal SSIM score
  - Optical Flow Consistency score

## Potential Improvements

- Integrate advanced global optimization heuristics (e.g., Simulated Annealing, Genetic Algorithms) for improved sequencing.
- Utilize multi-frame contextual embeddings via LSTM or Transformer models to capture temporal dynamics.
- Add motion-aware smoothness penalties to enforce temporal coherence.
- Implement video frame interpolation as a post-processing step to enhance visual smoothness.
- Accelerate feature extractions and similarity computations using GPU acceleration.

## Contact

For questions or contributions, please contact: [kaushiktanmay332@gmail.com](mailto:kaushiktanmay332@gmail.com).

