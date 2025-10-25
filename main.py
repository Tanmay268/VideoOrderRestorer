import os
import subprocess

def step(msg):
    print(f"\n{'='*10} {msg} {'='*10}\n")

def run_script(script, *args):
    cmd = ["python", script] + list(args)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    step("Extract Frames")
    run_script("extractFrames.py", "sample data/sample_video_1.mp4", "frames")

    step("Shuffle Frames (add your script if needed)")
    # run_script("shuffleFrames.py")  # Uncomment if needed

    step("Compute Similarity Matrix")
    run_script("computeSimilarity.py", "frames")

    step("Reconstruct Frame Order")
    run_script("reconstructOrder.py", "similarity_matrix.csv", "frame_order.txt")

    step("Convert to Video")
    run_script("convertToVideo.py", "frames", "frame_order.txt", "output_reconstructed.mp4")

    step("Evaluate Reconstruction")
    run_script("evaluate.py", "frames", "reconstructed_frames")

    step("Done!")
