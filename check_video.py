import cv2

cap = cv2.VideoCapture("jumbled_video.mp4")

if not cap.isOpened():
    print("Could not open video file.")
else:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Frame count: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    
    cap.release()
