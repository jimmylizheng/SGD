import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_metrics(dir1, dir2):
    # List all files in the directories
    files_dir1 = sorted([f for f in os.listdir(dir1) if f.startswith("screencapture-") and f.endswith(".png")])
    files_dir2 = sorted([f for f in os.listdir(dir2) if f.startswith("screencapture-") and f.endswith(".png")])
    
    if len(files_dir1) != len(files_dir2):
        print("Error: The number of images in dir1 and dir2 are not the same.")
        return
    
    total_ssim = 0
    total_psnr = 0
    count = 0
    
    for file1, file2 in zip(files_dir1, files_dir2):
        path1 = os.path.join(dir1, file1)
        path2 = os.path.join(dir2, file2)
        
        # Read images
        img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not read images {file1} or {file2}. Skipping.")
            continue
        
        if img1.shape != img2.shape:
            print(f"Warning: Image shapes do not match for {file1} and {file2}. Skipping.")
            continue
        
        # Determine a suitable window size for SSIM
        min_dim = min(img1.shape[:2])  # Minimum dimension of the image
        win_size = min(7, min_dim // 2 * 2 + 1)  # Use 7 or a smaller odd value
        
        # Compute SSIM
        # ssim_value = ssim(img1, img2, multichannel=True)
        ssim_value = ssim(img1, img2, win_size=win_size, multichannel=True, channel_axis=-1)
        
        # Compute PSNR
        psnr_value = cv2.PSNR(img1, img2)
        
        print(f"File: {file1} -> SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}")
        
        total_ssim += ssim_value
        total_psnr += psnr_value
        count += 1
    
    if count > 0:
        print(f"\nAverage SSIM: {total_ssim / count:.4f}")
        print(f"Average PSNR: {total_psnr / count:.4f}")
    else:
        print("No valid image pairs found.")

# Define directories
dir1 = "data/groundtruth"  # Replace with the path to dir1
dir2 = "data/"  # Replace with the path to dir2

# Call the function
compute_metrics(dir1, dir2)
