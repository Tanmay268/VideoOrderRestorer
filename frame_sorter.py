import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


class VideoFrameSorter:
    def __init__(self, input_video_path, output_video_path):
        self.input_video = input_video_path
        self.output_video = output_video_path
        self.frames = []
        self.frames_small = []  # Downsampled frames for faster processing
        self.frame_count = 0
        self.fps = 30
        self.frame_size = None

    def extract_frames(self):
        print("Extracting frames from video...")
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.input_video}")

        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                           )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
            frame_small = cv2.resize(frame, (320, 180))
            self.frames_small.append(frame_small)
        cap.release()
        self.frame_count = len(self.frames)
        print(f"Frames extracted: {self.frame_count} (FPS: {self.fps}, Frame size: {self.frame_size})")
        print("Processing at reduced resolution for speed.")

    def compute_frame_similarity_fast(self, idx1, idx2):
        f1 = self.frames_small[idx1]
        f2 = self.frames_small[idx2]
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        ssim_score = ssim(gray1, gray2)
        hist1 = cv2.calcHist([f1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([f2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        combined_score = (0.7 * ssim_score + 0.3 * hist_score)
        return combined_score

    def build_similarity_matrix_parallel(self):
        print("Building similarity matrix (parallel processing)...")
        n = self.frame_count
        similarity_matrix = np.zeros((n, n))
        comparisons = [(i, j) for i in range(n) for j in range(i + 1, n)]
        print(f"Similarity comparisons to compute: {len(comparisons)}")
        max_workers = min(8, os.cpu_count() or 4)
        print(f"Using {max_workers} threads.")
        count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(self.compute_frame_similarity_fast, i, j): (i, j)
                for i, j in comparisons
            }
            for future in as_completed(future_to_pair):
                i, j = future_to_pair[future]
                similarity = future.result()
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
                count += 1
                if count % 1000 == 0:
                    print(f"Computed {count} / {len(comparisons)} similarities")
        print("Similarity matrix construction complete.")
        return similarity_matrix

    def find_optimal_sequence_greedy(self, similarity_matrix):
        print("Finding optimal frame sequence using greedy algorithm.")
        n = self.frame_count
        visited = set()
        sequence = []
        avg_similarities = np.mean(similarity_matrix, axis=1)
        current = np.argmax(avg_similarities)
        sequence.append(current)
        visited.add(current)
        count = 0
        while len(visited) < n:
            unvisited = [i for i in range(n) if i not in visited]
            similarities = similarity_matrix[current][unvisited]
            best_idx = np.argmax(similarities)
            best_next = unvisited[best_idx]
            sequence.append(best_next)
            visited.add(best_next)
            current = best_next
            count += 1
            if count % 50 == 0:
                print(f"Sequence building progress: {count} / {n - 1}")
        print("Frame sequence determined.")
        return sequence

    def refine_sequence_fast(self, sequence, similarity_matrix):
        print("Refining sequence. Determining best starting frame...")
        n = len(sequence)
        sample_positions = range(0, n, max(1, n // 30))
        best_start = 0
        best_score = 0
        count = 0
        for start_idx in sample_positions:
            total_similarity = 0
            for i in range(min(n - 1, 50)):
                curr_frame = sequence[(start_idx + i) % n]
                next_frame = sequence[(start_idx + i + 1) % n]
                total_similarity += similarity_matrix[curr_frame][next_frame]
            if total_similarity > best_score:
                best_score = total_similarity
                best_start = start_idx
            count += 1
            if count % 5 == 0:
                print(f"Refinement progress: {count} / {len(sample_positions)} candidate starts tested")
        refined_sequence = sequence[best_start:] + sequence[:best_start]
        print("Sequence refinement complete.")
        return refined_sequence

    def write_output_video(self, sequence):
        print("Writing output video...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, self.fps, self.frame_size)
        for idx, frame_idx in enumerate(sequence):
            out.write(self.frames[frame_idx])
            if (idx + 1) % 50 == 0:
                print(f"Wrote {idx + 1} / {len(sequence)} frames to output.")
        out.release()
        print(f"Output video saved to: {self.output_video}")

    def calculate_accuracy(self, sequence):
        correct_positions = 0
        for position, frame_idx in enumerate(sequence):
            if position == frame_idx:
                correct_positions += 1
        accuracy = (correct_positions / len(sequence)) * 100
        return accuracy

    def neighbor_consistency(self, sequence):
        correct_pairs = 0
        for i in range(len(sequence) - 1):
            if sequence[i] + 1 == sequence[i + 1]:
                correct_pairs += 1
        return (correct_pairs / (len(sequence) - 1)) * 100

    def reconstruct(self):
        start_time = time.time()
        self.extract_frames()
        similarity_matrix = self.build_similarity_matrix_parallel()
        sequence = self.find_optimal_sequence_greedy(similarity_matrix)
        refined_sequence = self.refine_sequence_fast(sequence, similarity_matrix)
        self.write_output_video(refined_sequence)
        accuracy = self.calculate_accuracy(refined_sequence)
        neighbor_acc = self.neighbor_consistency(refined_sequence)
        end_time = time.time()
        execution_time = end_time - start_time
        print("-" * 60)
        print("Reconstruction completed.")
        print(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")
        print(f"Frame order accuracy (exact position): {accuracy:.2f}%")
        print(f"Neighbor pair accuracy: {neighbor_acc:.2f}%")
        print("-" * 60)
        return execution_time, accuracy, neighbor_acc


def main():
    INPUT_VIDEO = "jumbled_video.mp4"
    OUTPUT_VIDEO = "reconstructed_video.mp4"
    print("=" * 60)
    print("VIDEO FRAME RECONSTRUCTION (Student Version)")
    print("=" * 60)
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video '{INPUT_VIDEO}' not found!")
        print(f"Current directory: {os.getcwd()}")
        return
    print(f"Found input video: {INPUT_VIDEO}")
    print(f"File size: {os.path.getsize(INPUT_VIDEO) / (1024 * 1024):.2f} MB")
    try:
        sorter = VideoFrameSorter(INPUT_VIDEO, OUTPUT_VIDEO)
        execution_time, accuracy, neighbor_acc = sorter.reconstruct()
        with open("execution_time.log", "w") as f:
            f.write("Video Frame Reconstruction (Student Version)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Input: {INPUT_VIDEO}\n")
            f.write(f"Output: {OUTPUT_VIDEO}\n")
            f.write(f"Total frames: {sorter.frame_count}\n")
            f.write(f"FPS: {sorter.fps}\n")
            f.write(f"Resolution: {sorter.frame_size}\n")
            f.write(f"Execution time: {execution_time:.2f} seconds\n")
            f.write(f"Execution time: {execution_time / 60:.2f} minutes\n")
            f.write(f"Frame order accuracy (exact position): {accuracy:.2f}%\n")
            f.write(f"Neighbor pair accuracy: {neighbor_acc:.2f}%\n")
            f.write("\nOptimizations applied:\n")
            f.write("- Parallel processing (multi-threading)\n")
            f.write("- Downsampled frames (320x180)\n")
            f.write("- Fast greedy algorithm\n")
            f.write("- Sampled refinement\n")
        print("Execution time and accuracy logged to: execution_time.log")
    except Exception as e:
        print("\nError occurred during reconstruction:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
