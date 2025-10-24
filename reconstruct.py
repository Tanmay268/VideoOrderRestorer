import numpy as np
import os
import shutil

# Load the similarity matrix
similarity_matrix = np.loadtxt("similarity_matrix_parallel.csv", delimiter=",")
num_frames = similarity_matrix.shape[0]

# Track the order of frames
used = set()
order = []

# Start from frame 0 (or pick np.argmin(similarity_matrix.sum(axis=1)) for best starting point)
current = 0
order.append(current)
used.add(current)

for _ in range(num_frames - 1):
    # Set high value for already-used frames to avoid selecting them
    similarities = similarity_matrix[current].copy()
    for idx in used:
        similarities[idx] = np.inf
    # Select the frame with the lowest difference
    next_frame = np.argmin(similarities)
    order.append(next_frame)
    used.add(next_frame)
    current = next_frame

print("Frame order reconstructed:", order)

# Step 2: OPTIONAL — Copy frames to 'reconstructed_frames' in correct order
input_folder = "D:/Downloads/Tecdia/shuffled_frames"
output_folder = "D:/Downloads/Tecdia/reconstructed_frames"
os.makedirs(output_folder, exist_ok=True)
frame_files = sorted(os.listdir(input_folder))

for idx, frame_index in enumerate(order):
    src = os.path.join(input_folder, frame_files[frame_index])
    dst = os.path.join(output_folder, f"frame_{idx:04d}.png")
    shutil.copy(src, dst)

print("Frames saved in reconstructed order.")
