import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot the distribution of the Gaussian data

# Load data from JSON file
with open('gaussian_data.json', 'r') as f:
    data = json.load(f)

# Convert lists to numpy arrays for easier handling
positions = np.array(data['positions']).reshape(-1, 3)
opacities = np.array(data['opacities'])
colors = np.array(data['colors']).reshape(-1, 3)
cov3Ds = np.array(data['cov3Ds']).reshape(-1, 6)

# Function to save plot
def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

# Function to calculate CDF
def calculate_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf

# 1. Plot and save the CDF of positions (X, Y, Z coordinates)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for i, label, color in zip(range(3), ['X Position', 'Y Position', 'Z Position'], ['r', 'g', 'b']):
#     sorted_data, cdf = calculate_cdf(positions[:, i])
#     axes[i].plot(sorted_data, cdf, color=color, label=label)
#     axes[i].legend()

# fig.suptitle('CDF of Positions')
# save_plot(fig, 'positions_cdf.png')

# 2. Plot and save the CDF of opacities
fig = plt.figure(figsize=(8, 6))
sorted_opacities, cdf_opacities = calculate_cdf(opacities)
plt.plot(sorted_opacities, cdf_opacities, color='purple')
plt.title('CDF of Opacities')
plt.xlabel('Opacity')
save_plot(fig, 'opacities_cdf.png')

# 3. Plot and save the CDF of colors (R, G, B channels)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for i, label, color in zip(range(3), ['Red Channel', 'Green Channel', 'Blue Channel'], ['r', 'g', 'b']):
#     sorted_data, cdf = calculate_cdf(colors[:, i])
#     axes[i].plot(sorted_data, cdf, color=color, label=label)
#     axes[i].legend()

# fig.suptitle('CDF of Colors')
# save_plot(fig, 'colors_cdf.png')

# 4. Plot and save the CDF of covariance matrix elements
# fig = plt.figure(figsize=(10, 6))
# for i, label, color in zip(range(2), ['Covariance Element 1', 'Covariance Element 2'], ['blue', 'orange']):
#     sorted_data, cdf = calculate_cdf(cov3Ds[:, i])
#     plt.plot(sorted_data, cdf, color=color, label=label)

# plt.title('CDF of Covariance Matrix Elements')
# plt.legend()
# save_plot(fig, 'covariance_cdf.png')