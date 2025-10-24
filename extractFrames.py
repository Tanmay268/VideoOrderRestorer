import cv2
import os

video_path = "D:/Downloads/Tecdia/sample data/sample_video_1.mp4"  # Update with your video file path
output_folder = "rain_drops_frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.png", frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames.")
