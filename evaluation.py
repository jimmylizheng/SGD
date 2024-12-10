import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_ssim_psnr(dir1, dir2, a, b, file_extension=".png", is_qoe=False):
    ssim_values = []
    psnr_values = []

    for n in range(a, b + 1):
        filename = f"screencapture-{n}{file_extension}"
        file1_path = os.path.join(dir1, filename)
        file2_path = os.path.join(dir2, filename)

        # Ensure both files exist
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            print(f"Skipping {filename}: File not found in one of the directories.")
            continue

        # Load images
        img1 = cv2.imread(file1_path)
        img2 = cv2.imread(file2_path)

        # Ensure images are valid and of the same size
        if img1 is None or img2 is None:
            print(f"Skipping {filename}: Unable to read one of the files.")
            continue
        if img1.shape != img2.shape:
            print(f"Skipping {filename}: Images have different shapes.")
            continue

        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM and PSNR
        ssim_value = ssim(gray1, gray2)
        psnr_value = psnr(img1, img2)

        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)

        # print(f"Processed {filename}: SSIM = {ssim_value:.4f}, PSNR = {psnr_value:.4f}")

    for index, value in enumerate(psnr_values):
        if np.isinf(value):
            i = index + 1
            while i<len(psnr_values) and np.isinf(psnr_values[i]):
                i=i+1
            for j in range(index, i):
                psnr_values[j]=(psnr_values[index-1]+psnr_values[i])/2
        
    if is_qoe:
        for i in range(len(ssim_values)):
            ssim_values[i]=ssim_values[i]/(a+i)
        for i in range(len(psnr_values)):
            psnr_values[i]=psnr_values[i]/(a+i)
        out_ssim = np.sum(ssim_values) if ssim_values else 0
        out_psnr = np.sum(psnr_values) if psnr_values else 0
    else:
        # Calculate averages
        out_ssim = np.mean(ssim_values) if ssim_values else 0
        out_psnr = np.mean(psnr_values) if psnr_values else 0

    return out_ssim, out_psnr


# Directories and range
directory1 = "data/gt" # groundtruth directory
directory2 = "data/"

dir_list = ["na_1_10", "na_10_1", "na_1_30", "na_10_3", "br_10_1", "br_10_3", 
            "op_10_1", "op_10_3", "sp_10_1", "sp_10_3", ]

starting_time = 10  # Replace with your start value
ending_time = 55  # Replace with your end value
file_ext = ".png"  # Change file extension if needed (e.g., ".jpg")

for tmp_dir in dir_list:
    print("\nProcessing directory: ",tmp_dir)
    # Compute averages
    result_ssim, result_psnr = compute_ssim_psnr(directory1, directory2+tmp_dir, starting_time, ending_time, file_ext, False)

    print(f"\nAverage SSIM: {result_ssim:.4f}")
    print(f"Average PSNR: {result_psnr:.4f}")

    # Compute averages
    result_ssim, result_psnr = compute_ssim_psnr(directory1, directory2+tmp_dir, starting_time, ending_time, file_ext, True)

    print(f"\nQoE(SSIM): {result_ssim:.4f}")
    print(f"QoE(PSNR): {result_psnr:.4f}")

