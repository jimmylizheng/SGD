import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot the distribution of the Gaussian data

# Load data from JSON file
with open('gaussian_data.json', 'r') as f:
    data = json.load(f)

# Convert lists to numpy arrays for easier handling
# positions = np.array(data['positions']).reshape(-1, 3)
color_sum = np.array(data['color_sum'])
brightness = np.array(data['brightness'])
scale_mul = np.array(data['scale_mul'])
scale_op_mul = np.array(data['scale_op_mul'])
opacities = np.array(data['opacities'])
# colors = np.array(data['colors']).reshape(-1, 3)
# cov3Ds = np.array(data['cov3Ds']).reshape(-1, 6)

# Function to save plot
def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

# Function to calculate CDF
def calculate_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf

# fig = plt.figure(figsize=(8, 6))
# sorted_color_sum, cdf_color_sum = calculate_cdf(color_sum)
# plt.plot(sorted_color_sum, cdf_color_sum, color='purple')
# # plt.title('CDF of color_sum')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('color_sum', fontsize=22)
# save_plot(fig, 'result/color_sum_cdf.png')

fig = plt.figure(figsize=(9, 8))
sorted_brightness, cdf_brightness = calculate_cdf(brightness)
plt.plot(sorted_brightness, cdf_brightness, color='blue', linewidth=2)
# plt.title('CDF of brightness')
plt.xlabel('brightness', fontsize=22)
plt.xlim(0, 1.75)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
save_plot(fig, 'result/brightness_cdf.png')

fig = plt.figure(figsize=(9, 8))
sorted_scale_mul, cdf_scale_mul = calculate_cdf(scale_mul)
plt.plot(sorted_scale_mul, cdf_scale_mul, color='blue', linewidth=2)
# plt.title('CDF of scale_mul')
plt.xlabel('Scale', fontsize=22)
# plt.xlim(0, 0.01)
plt.xlim(1e-10, 1)
plt.xscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
save_plot(fig, 'result/scale_mul_cdf.png')

# fig = plt.figure(figsize=(8, 6))
# sorted_scale_op_mul, cdf_scale_op_mul = calculate_cdf(scale_op_mul)
# plt.plot(sorted_scale_op_mul, cdf_scale_op_mul, color='purple')
# # plt.title('CDF of scale_op_mul')
# plt.xlabel('scale_op_mul')
# # plt.xlim(0, 0.001)
# plt.xlim(1e-10, 1)
# plt.xscale('log')
# save_plot(fig, 'result/scale_op_mul_cdf.png')

fig = plt.figure(figsize=(9, 8))
sorted_opacities, cdf_opacities = calculate_cdf(opacities)
plt.plot(sorted_opacities, cdf_opacities, color='blue', linewidth=2)
# plt.title('CDF of Opacities')
plt.xlabel('Opacity', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
save_plot(fig, 'result/opacities_cdf.png')
