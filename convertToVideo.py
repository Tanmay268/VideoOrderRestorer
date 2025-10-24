import cv2
import os

frames_folder = "D:/Downloads/Tecdia/reconstructed_frames"
output_video = "D:/Downloads/Tecdia/reconstructed_video.mp4"
fps = 30

frame_files = sorted(os.listdir(frames_folder))
frame_paths = [os.path.join(frames_folder, f) for f in frame_files]

frame = cv2.imread(frame_paths[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for idx, frame_path in enumerate(frame_paths):
    frame = cv2.imread(frame_path)
    if frame is not None:
        video_writer.write(frame)
        print(f"Added frame: {frame_path}")
    else:
        print(f"Failed to read: {frame_path}")

video_writer.release()
print("Finished writing video:", output_video)
