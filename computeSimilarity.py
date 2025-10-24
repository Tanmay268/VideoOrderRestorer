import cv2
import numpy as np
import os

def compute_difference(img1, img2):
    diff = np.sum(np.abs(img1.astype(int) - img2.astype(int)))
    return diff

frames_folder = "D:/Downloads/Tecdia/shuffled_frames"
frame_files = sorted(os.listdir(frames_folder))

images = [cv2.imread(os.path.join(frames_folder, f)) for f in frame_files]

num_frames = len(images)
similarity_matrix = np.zeros((num_frames, num_frames))

for i in range(num_frames):
    for j in range(num_frames):
        if i != j:
            similarity_matrix[i, j] = compute_difference(images[i], images[j])
        else:
            similarity_matrix[i, j] = 0  # No difference with self

print(similarity_matrix)
# Optionally, save matrix to file
np.savetxt("similarity_matrix.csv", similarity_matrix, delimiter=",")
