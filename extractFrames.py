import cv2
import os
import sys

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{count:04d}.png"), frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames to {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extractFrames.py input_video output_folder")
        sys.exit(1)
    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    extract_frames(video_path, output_folder)
