# Video Frame Reconstruction Challenge

## Overview
This project reconstructs a jumbled video by analyzing frame similarities and determining the optimal sequential order.

## Algorithm Approach

### Method: Hybrid Similarity-Based Greedy Reconstruction

**Core Strategy:**
1. **Frame Extraction:** Extract all frames from the input jumbled video.
2. **Similarity Matrix:** Compute pairwise frame similarity scores using multiple measures:
   - Structural Similarity Index (SSIM) - 30% weight
   - Color Histogram correlation - 15% weight
   - Edge Histogram correlation - 10% weight
   - Optical Flow motion consistency - 10% weight
   - CNN feature similarity (MobileNetV2) - 40% weight
3. **Greedy Nearest Neighbor:** Build the frame sequence starting from multiple candidate frames (multi-start) by connecting most similar consecutive frames.
4. **Sequence Refinements:** Optimize starting point and ordering using:
   - Sequence reversal check to choose best directional order
   - 2-opt local search optimizations
   - Windowed 2-opt optimizations for fine-grained improvement

### Why This Approach?

- **SSIM:** Captures fine structural changes between frames.
- **Histogram Correlation:** Ensures color and lighting consistency.
- **Optical Flow:** Encodes motion continuity between frames.
- **Deep CNN Features:** Provides rich semantic frame representations.
- **Multi-Start Greedy:** Avoids poor local minima by trying several initial frame choices.
- **Local Search:** Refines ordering to maximize overall similarity.

### Key Design Considerations

- **Accuracy:** Multi-feature similarity robustly captures frame relationships.
- **Performance:** Downsamples frames to 320x180 (approx 25% of original pixels) for faster processing.
- **Scalability:** Runs efficiently on typical laptops with 16GB RAM and no GPU.
- **Parallelization:** Uses multithreading to compute pairwise similarities concurrently.
- **Simplicity:** Uses well-understood greedy and local search heuristics suitable for NP-hard ordering problem.

### Time Complexity

| Step                  | Complexity |
|-----------------------|------------|
| Frame extraction      | O(n)       |
| Similarity matrix     | O(n²)      |
| Sequence initialization | O(n²)    |
| 2-opt local search    | O(k·n²) (k: max ops) |
| Total                 | O(n²)      |

where n = number of frames (~300).

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download this repository:
git clone <repository_url>
cd

2. Install dependencies:
pip install opencv-python numpy scikit-image torch torchvision scipy


---

## Usage

1. Place your jumbled video file named `jumbled_video.mp4` in the project root directory.

2. Run the reconstruction pipeline:
   python frame_sorter.py


3. The output reconstructed video will be saved as `reconstructed_video.mp4`.

4. Execution time and accuracy metrics will be logged to `execution_time.log`.

---

## Logging and Output

- `execution_time.log` will include input/output file info, frame count, FPS, resolution, execution duration, and accuracy metrics.
- Accuracy metrics include:
  - Exact frame order accuracy (% frames in correct position).
  - Neighbor frame accuracy (% consecutive frame pairs correctly ordered).

---

## Potential Improvements

- Incorporate advanced optimization algorithms (Simulated Annealing, Genetic Algorithms).
- Use multi-frame contextual embeddings via LSTM or Transformer models.
- Implement motion-aware smoothness penalties for temporal consistency.
- Apply video interpolation post-processing to smooth visual transitions.

---

## Contact

For questions or contributions, please contact - kaushiktanmay332@gmail.com.



