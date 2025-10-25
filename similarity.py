from skimage.metrics import structural_similarity as compare_ssim
import cv2
import numpy as np

def pixel_difference(img1, img2):
    return np.sum(np.abs(img1.astype(int) - img2.astype(int)))

def ssim_difference(img1, img2):
    score, _ = compare_ssim(img1, img2, full=True)
    return 1 - score

def orb_feature_distance(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return float('inf')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) == 0:
        return float('inf')
    return 1000 / len(matches)  # Lower score = higher similarity

def combined_similarity(img1, img2):
    pdiff = pixel_difference(img1, img2)
    sdiff = ssim_difference(img1, img2) * 1000
    odiff = orb_feature_distance(img1, img2)
    return pdiff + sdiff + odiff
