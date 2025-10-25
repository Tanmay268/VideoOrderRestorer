import os
import numpy as np
import cv2
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def compute_pair_similarity(args, folder, files):
    i, j = args
    img1 = cv2.imread(os.path.join(folder, files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(folder, files[j]), cv2.IMREAD_GRAYSCALE)
    
    # Downsample images for faster computation
    img1 = cv2.resize(img1, (160, 90))
    img2 = cv2.resize(img2, (160, 90))
    
    # Use only pixel difference (fastest)
    diff = np.sum(np.abs(img1.astype(int) - img2.astype(int)))
    
    return (i, j, diff)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python computeSimilarity.py frames_folder")
        sys.exit(1)

    folder = sys.argv[1]
    files = sorted(os.listdir(folder))
    n = len(files)
    similarity_matrix = np.full((n, n), np.inf, dtype=np.float32)

    # Generate pairs efficiently
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    
    # Use partial to avoid passing large data
    compute_func = partial(compute_pair_similarity, folder=folder, files=files)

    print(f"Computing similarity for {n} frames ({len(pairs)} pairs)...")
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(compute_func, pairs, chunksize=100)
        for i, j, score in results:
            similarity_matrix[i, j] = score
            if (i * n + j) % 10000 == 0:
                print(f"Processed {i * n + j}/{len(pairs)} pairs...")

    np.savetxt("similarity_matrix.csv", similarity_matrix, delimiter=",")
    print("Similarity matrix saved to similarity_matrix.csv")
