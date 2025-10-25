import cv2
import numpy as np
import os
import sys

def create_video(frames_folder, order_file, output_file, fps=30):
    order = np.loadtxt(order_file, dtype=int)
    files = sorted(os.listdir(frames_folder))
    frame_example = cv2.imread(os.path.join(frames_folder, files[0]))
    h, w, _ = frame_example.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    for idx in order:
        frame = cv2.imread(os.path.join(frames_folder, files[idx]))
        video_writer.write(frame)

    video_writer.release()
    print(f"Reconstructed video saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convertToVideo.py frames_folder order_file output_video.mp4")
        sys.exit(1)
    create_video(sys.argv[1], sys.argv[2], sys.argv[3])
