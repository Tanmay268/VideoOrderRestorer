import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import models, transforms
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, time
from sklearn.cluster import KMeans
from itertools import permutations


class VideoFrameSorter:
    def __init__(self, input_video_path, output_video_path):
        self.input_video = input_video_path
        self.output_video = output_video_path
        self.frames = []
        self.frames_small = []
        self.features = []
        self.frame_count = 0
        self.fps = 30
        self.frame_size = None

        self.W_SSIM = 0.25
        self.W_HIST = 0.15
        self.W_CNN = 0.40
        self.W_EDGE = 0.10
        self.W_FLOW = 0.10

        cnn = models.mobilenet_v2(pretrained=True)
        self.feat_extractor = torch.nn.Sequential(*list(cnn.children())[:-1])
        self.feat_extractor.eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self):
        print("Extracting frames from video...")
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.input_video}")
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
            self.frames_small.append(cv2.resize(frame, (320, 180)))
        cap.release()
        self.frame_count = len(self.frames)
        print(f"Frames extracted: {self.frame_count}")

    def compute_cnn_features(self):
        print("Extracting CNN features (MobileNetV2, CPU)...")
        features = []
        with torch.no_grad():
            for idx, frame in enumerate(self.frames):
                img = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
                input_tensor = self.preprocess(img)
                input_tensor = input_tensor.unsqueeze(0)
                feat = self.feat_extractor(input_tensor).flatten().numpy()
                features.append(feat)
                if (idx + 1) % 25 == 0:
                    print(f"Extracted features: {idx + 1}/{self.frame_count}")
        self.features = np.array(features)
        print("CNN feature extraction complete.")

    def edge_histogram(self, img):
        edges = cv2.Canny(img, 50, 150)
        hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def optical_flow_similarity(self, f1, f2):
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = np.mean(mag)
        return -mean_mag

    def compute_hybrid_similarity(self):
        print("Computing hybrid similarity matrix (parallel)...")
        n = self.frame_count
        ssim_arr = np.zeros((n, n))
        hist_arr = np.zeros((n, n))
        edge_arr = np.zeros((n, n))
        flow_arr = np.zeros((n, n))
        comparisons = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def worker(i, j):
            f1, f2 = self.frames_small[i], self.frames_small[j]
            gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            ssim_score = ssim(gray1, gray2)
            h1 = cv2.calcHist([f1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            h2 = cv2.calcHist([f2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            h1 = cv2.normalize(h1, h1).flatten()
            h2 = cv2.normalize(h2, h2).flatten()
            hist_score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            e1 = self.edge_histogram(f1)
            e2 = self.edge_histogram(f2)
            edge_score = cv2.compareHist(e1, e2, cv2.HISTCMP_CORREL)
            flow_score = self.optical_flow_similarity(f1, f2)
            return (i, j, ssim_score, hist_score, edge_score, flow_score)

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(worker, i, j) for i, j in comparisons]
            for count, f in enumerate(as_completed(futures), 1):
                i, j, ssim_score, hist_score, edge_score, flow_score = f.result()
                ssim_arr[i, j] = ssim_arr[j, i] = ssim_score
                hist_arr[i, j] = hist_arr[j, i] = hist_score
                edge_arr[i, j] = edge_arr[j, i] = edge_score
                flow_arr[i, j] = flow_arr[j, i] = flow_score
                if count % 1000 == 0:
                    print(f"Hybrid features done: {count}/{len(comparisons)}")

        print("Image features done; computing CNN similarity...")
        sim_feat_matrix = 1 - cdist(self.features, self.features, metric='cosine')
        sim_matrix = (self.W_SSIM * ssim_arr
                      + self.W_HIST * hist_arr
                      + self.W_CNN * sim_feat_matrix
                      + self.W_EDGE * edge_arr
                      + self.W_FLOW * flow_arr)
        return sim_matrix

    def find_sequence_greedy_multi_start(self, similarity_matrix, n_starts=5):
        print(f"Running greedy multi-start assembly ({n_starts} starts)...")
        n = self.frame_count
        avg_sim = np.mean(similarity_matrix, axis=1)
        seeds = np.argsort(-avg_sim)[:n_starts]
        best_score = -np.inf
        best_sequence = None

        for seed in seeds:
            visited = set([seed])
            sequence = [seed]
            current = seed
            while len(visited) < n:
                unvisited = [i for i in range(n) if i not in visited]
                sims = similarity_matrix[current][unvisited]
                next_frame = unvisited[np.argmax(sims)]
                sequence.append(next_frame)
                visited.add(next_frame)
                current = next_frame
            score = sum(similarity_matrix[sequence[i]][sequence[i + 1]] for i in range(n - 1))
            print(f"Seed {seed} score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_sequence = sequence
        print("Best sequence selected from multi-start.")
        return best_sequence

    def choose_best_direction(self, sequence, similarity_matrix):
        score_fwd = sum(similarity_matrix[sequence[i]][sequence[i + 1]] for i in range(len(sequence) - 1))
        rev_seq = sequence[::-1]
        score_rev = sum(similarity_matrix[rev_seq[i]][rev_seq[i + 1]] for i in range(len(rev_seq) - 1))
        if score_rev > score_fwd:
            print("Using reversed sequence for better total similarity.")
            return rev_seq
        else:
            return sequence

    def two_opt(self, sequence, similarity_matrix, max_iter=15):
        print(f"Performing 2-opt local search (max_iter={max_iter})...")
        n = len(sequence)
        best = sequence[:]
        improved = True
        count = 0

        def path_score(seq):
            return sum(similarity_matrix[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))

        best_score = path_score(best)
        while improved and count < max_iter:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    new_seq = best[:i] + best[i:j][::-1] + best[j:]
                    new_score = path_score(new_seq)
                    if new_score > best_score:
                        best = new_seq
                        best_score = new_score
                        improved = True
                        print(f"2-opt improvement iteration {count + 1} flipping {i}-{j}")
            count += 1
        print("2-opt search complete.")
        return best

    def windowed_two_opt(self, sequence, similarity_matrix, window=30, n_passes=2):
        print(f"Starting windowed 2-opt: window={window}, passes={n_passes}...")
        n = len(sequence)
        best = sequence[:]
        for _ in range(n_passes):
            for start in range(0, n - window + 1, window // 2):
                wseq = best[start:start + window]
                improved_seq = self.two_opt(wseq, similarity_matrix, max_iter=3)
                best[start:start + window] = improved_seq
        print("Windowed 2-opt complete.")
        return best

    def write_output_video(self, sequence):
        print("Writing output video...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, self.fps, self.frame_size)
        for idx, frame_idx in enumerate(sequence):
            out.write(self.frames[frame_idx])
            if (idx + 1) % 50 == 0:
                print(f"Wrote {idx + 1} / {len(sequence)} frames")
        out.release()

    def calculate_smoothness(self, sequence, similarity_matrix):
        consecutive_similarities = [similarity_matrix[sequence[i], sequence[i + 1]] for i in range(len(sequence) - 1)]
        smoothness_score = np.mean(consecutive_similarities) * 100
        return smoothness_score

    def temporal_ssim(self, frames):
        n = len(frames)
        scores = []
        for i in range(n - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            score, _ = ssim(gray1, gray2, full=True)
            scores.append(score)
        return np.mean(scores) * 100

    def refined_optical_flow_consistency(self, frames):
        angles = []
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            angles.append(ang.flatten())
        all_angles = np.concatenate(angles)
        angle_variance = np.var(all_angles)
        return 100 / (1 + angle_variance)

    def check_and_fix_direction_consistency(self, sequence):
        initial_angles = []
        for i in range(1, min(10, len(sequence))):
            gray1 = cv2.cvtColor(self.frames[sequence[i - 1]], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.frames[sequence[i]], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            initial_angles.append(np.median(ang))
        avg_initial_angle = np.mean(initial_angles)

        total_angles = []
        for i in range(1, len(sequence)):
            gray1 = cv2.cvtColor(self.frames[sequence[i - 1]], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.frames[sequence[i]], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            total_angles.append(np.median(ang))
        avg_total_angle = np.mean(total_angles)

        angle_diff = np.abs(avg_total_angle - avg_initial_angle)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        threshold = np.pi / 2
        if angle_diff > threshold:
            print("Detected direction flip based on optical flow; reversing sequence.")
            return sequence[::-1]
        return sequence

    def assign_clusters(self, n_clusters=3):
        print(f"Assigning frames to {n_clusters} clusters using KMeans...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(self.features)
        print("Cluster assignments complete.")
        return cluster_assignments

    def fix_cluster_directions(self, cluster_frames):
        print("Fixing direction inside each cluster...")
        fixed_clusters = []
        for frames_seq in cluster_frames:
            if len(frames_seq) <= 1:
                fixed_clusters.append(frames_seq)
                continue

            initial_angles = []
            for i in range(1, min(10, len(frames_seq))):
                gray1 = cv2.cvtColor(self.frames[frames_seq[i - 1]], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(self.frames[frames_seq[i]], cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                initial_angles.append(np.median(ang))
            avg_initial_angle = np.mean(initial_angles)

            total_angles = []
            for i in range(1, len(frames_seq)):
                gray1 = cv2.cvtColor(self.frames[frames_seq[i - 1]], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(self.frames[frames_seq[i]], cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                total_angles.append(np.median(ang))
            avg_total_angle = np.mean(total_angles)
            angle_diff = np.abs(avg_total_angle - avg_initial_angle)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            threshold = np.pi / 2

            if angle_diff > threshold:
                print(f"Reversing cluster with {len(frames_seq)} frames due to direction flip")
                fixed_clusters.append(frames_seq[::-1])
            else:
                fixed_clusters.append(frames_seq)
        return fixed_clusters

    def check_transition_discontinuity(self, idx1, idx2, threshold=30):
        f1 = self.frames[idx1]
        f2 = self.frames[idx2]
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = np.mean(mag)

        moments1 = cv2.moments(cv2.Canny(gray1, 50, 150))
        moments2 = cv2.moments(cv2.Canny(gray2, 50, 150))
        if moments1['m00'] != 0 and moments2['m00'] != 0:
            cx1 = int(moments1['m10'] / moments1['m00'])
            cy1 = int(moments1['m01'] / moments1['m00'])
            cx2 = int(moments2['m10'] / moments2['m00'])
            cy2 = int(moments2['m01'] / moments2['m00'])
            pos_diff = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        else:
            pos_diff = 0

        discontinuity_score = mean_mag + pos_diff
        return discontinuity_score > threshold

    def fix_cluster_boundary_transitions(self, cluster_frames):
        print("Fixing cluster boundary transitions...")
        changed = True
        while changed:
            changed = False
            for i in range(len(cluster_frames) - 1):
                last_frame_current = cluster_frames[i][-1]
                first_frame_next = cluster_frames[i + 1][0]

                if self.check_transition_discontinuity(last_frame_current, first_frame_next):
                    flipped = cluster_frames[i + 1][::-1]

                    old_discontinuity = self.check_transition_discontinuity(last_frame_current, first_frame_next)
                    new_discontinuity = self.check_transition_discontinuity(last_frame_current, flipped[0])

                    if new_discontinuity < old_discontinuity:
                        cluster_frames[i + 1] = flipped
                        changed = True
                        print(f"Flipped cluster {i + 1} to improve boundary transition.")
        return cluster_frames

    def find_best_cluster_order(self, cluster_assignments, similarity_matrix, n_clusters):
        print("Finding best cluster order by testing permutations...")
        cluster_frames = []
        for i in range(n_clusters):
            cluster_frames.append([j for j in range(self.frame_count) if cluster_assignments[j] == i])

        cluster_frames = self.fix_cluster_directions(cluster_frames)

        all_orders = list(permutations(range(n_clusters)))
        best_score = -np.inf
        best_order = None
        best_sequence = None

        for order in all_orders:
            sequence = []
            for cidx in order:
                sequence.extend(cluster_frames[cidx])
            score = self.calculate_smoothness(sequence, similarity_matrix)
            print(f"Order {order}, Smoothness Score: {score:.2f}%")
            if score > best_score:
                best_score = score
                best_order = order
                best_sequence = sequence

        # Fix boundary transitions between clusters
        ordered_clusters = [cluster_frames[i] for i in best_order]
        ordered_clusters = self.fix_cluster_boundary_transitions(ordered_clusters)

        final_sequence = []
        for cl in ordered_clusters:
            final_sequence.extend(cl)

        return final_sequence

    def reconstruct(self):
        total_start = time.time()
        self.extract_frames()
        self.compute_cnn_features()
        sim_matrix = self.compute_hybrid_similarity()

        n_clusters = 3
        cluster_assignments = self.assign_clusters(n_clusters=n_clusters)

        cluster_ordered_sequence = self.find_best_cluster_order(cluster_assignments, sim_matrix, n_clusters)

        init_seq = self.choose_best_direction(cluster_ordered_sequence, sim_matrix)
        refined_seq = self.two_opt(init_seq, sim_matrix, max_iter=15)
        windowed_seq = self.windowed_two_opt(refined_seq, sim_matrix, window=30, n_passes=2)

        consistent_seq = self.check_and_fix_direction_consistency(windowed_seq)

        self.write_output_video(consistent_seq)
        smoothness = self.calculate_smoothness(consistent_seq, sim_matrix)

        reordered_frames = [self.frames[i] for i in consistent_seq]
        temporal_ssim_score = self.temporal_ssim(reordered_frames)
        flow_consistency_score = self.refined_optical_flow_consistency(reordered_frames)

        elapsed = time.time() - total_start
        print("-" * 60)
        print("Reconstruction complete.")
        print(f"Execution time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        print(f"Sequence smoothness score: {smoothness:.2f}%")
        print(f"Temporal SSIM score: {temporal_ssim_score:.2f}%")
        print(f"Optical Flow Consistency score: {flow_consistency_score:.2f}%")
        print("-" * 60)
        return elapsed, smoothness, temporal_ssim_score, flow_consistency_score


def main():
    input_video = "jumbled_video.mp4"
    output_video = "reconstructed_video.mp4"
    print("=" * 60)
    print("VIDEO FRAME RECONSTRUCTION: Optical Flow + Hybrid Similarity + Multi-Greedy + 2-opt + Cluster Ordering + Boundary Fix")
    print("=" * 60)
    if not os.path.exists(input_video):
        print(f"Input video '{input_video}' not found in {os.getcwd()}")
        return
    sorter = VideoFrameSorter(input_video, output_video)
    exec_time, smooth_acc, temporal_ssim_score, flow_consistency_score = sorter.reconstruct()
    with open("execution_time.log", "w") as f:
        f.write("Video Frame Reconstruction Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input   : {input_video}\n")
        f.write(f"Output  : {output_video}\n")
        f.write(f"Frames  : {sorter.frame_count}\n")
        f.write(f"FPS     : {sorter.fps}\n")
        f.write(f"Resolution: {sorter.frame_size}\n")
        f.write(f"Execution Time           : {exec_time:.2f} seconds\n")
        f.write(f"Sequence Smoothness Score: {smooth_acc:.2f}%\n")
        f.write(f"Temporal SSIM Score      : {temporal_ssim_score:.2f}%\n")
        f.write(f"Optical Flow Consistency : {flow_consistency_score:.2f}%\n")
    print("Execution time and accuracy logged to execution_time.log")


if __name__ == "__main__":
    main()
