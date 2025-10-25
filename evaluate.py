import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import sys

def compute_ssim(img1, img2):
    ssim_score, _ = compare_ssim(img1, img2, full=True)
    return ssim_score

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def evaluate_videos(original_video, reconstructed_video):
    cap_orig = cv2.VideoCapture(original_video)
    cap_rec = cv2.VideoCapture(reconstructed_video)

    ssim_values = []
    psnr_values = []
    frame_num = 0

    while True:
        ret1, orig_frame = cap_orig.read()
        ret2, rec_frame = cap_rec.read()
        if not ret1 or not ret2:
            break

        orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        rec_gray = cv2.cvtColor(rec_frame, cv2.COLOR_BGR2GRAY)
        ssim_val = compute_ssim(orig_gray, rec_gray)
        psnr_val = compute_psnr(orig_gray, rec_gray)
        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)
        frame_num += 1

    cap_orig.release()
    cap_rec.release()

    print(f"Compared {frame_num} frames.")
    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    return avg_ssim, avg_psnr

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py original_video.mp4 reconstructed_video.mp4")
        sys.exit(1)
    evaluate_videos(sys.argv[1], sys.argv[2])
