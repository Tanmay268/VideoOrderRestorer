import os
import random
import shutil

input_folder = "rain_drops_frames"
shuffled_folder = "shuffled_frames"
os.makedirs(shuffled_folder, exist_ok=True)

# Get all frame files
frames = sorted(os.listdir(input_folder))
random.shuffle(frames)

# Copy frames into shuffled folder in random order
for idx, frame_file in enumerate(frames):
    src = os.path.join(input_folder, frame_file)
    dst = os.path.join(shuffled_folder, f"frame_{idx:04d}.png")
    shutil.copy(src, dst)

print(f"Shuffled and copied {len(frames)} frames.")
