import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def compute_difference_pair(args):
    i, j, img1, img2 = args
    diff = np.sum(np.abs(img1 - img2))  # Compare grayscale
    return (i, j, diff)

def load_images(folder):
    files = sorted(os.listdir(folder))
    images = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images, files

if __name__ == "__main__":
    frames_folder = "D:/Downloads/Tecdia/shuffled_frames"
    images, frame_files = load_images(frames_folder)
    num_frames = len(images)
    similarity_matrix = np.zeros((num_frames, num_frames), dtype=np.float32)

    # Prepare all unique pairs (i, j) where i != j
    args = [(i, j, images[i], images[j]) for i in range(num_frames) for j in range(num_frames) if i != j]
    cpu_cores = multiprocessing.cpu_count()

    # Use concurrent.futures for process-based parallelism
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = [executor.submit(compute_difference_pair, pair) for pair in args]
        for future in as_completed(futures):
            i, j, diff = future.result()
            similarity_matrix[i, j] = diff

    np.savetxt("similarity_matrix_parallel.csv", similarity_matrix, delimiter=",")
    print(f"Similarity matrix ({num_frames}x{num_frames}) computed and saved.")
