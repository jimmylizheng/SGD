import matplotlib.pyplot as plt

# Sample data
categories = ['Original', 'Size', 'Opacity', 'Brightness']
# 294.68MB, b1=29.47MB/s, b2=9.82MB/s
b1_SSIM_set=[0.6364, 0.6937, 0.6025, 0.6322] # for avg SSIM
b2_SSIM_set=[0.6046, 0.6098, 0.6376, 0.5450]

b1_qoe_SSIM_set=[0.8969, 1.0767, 0.8336, 0.8755] # for qoe SSIM
b2_qoe_SSIM_set=[0.8884, 0.8573, 0.9577, 0.8318]

b1_PSNR_set=[20.3495, 24.1375, 20.6008, 21.4434] # for avg PSNR
b2_PSNR_set=[19.2069, 19.6989, 21.0800, 14.9727]

b1_qoe_PSNR_set=[30.0078, 35.4656, 30.4252, 30.6310] # for qoe PSNR
b2_qoe_PSNR_set=[29.3344, 29.4198, 32.5110, 24.4323]

# Bar width
bar_width = 0.4

# Positions of the bars
positions1 = range(len(categories))
positions2 = [p + bar_width for p in positions1]

plt.figure(figsize=(10, 6))

# Create the bar plots
# plt.bar(positions1, b1_SSIM_set, width=bar_width, color='skyblue', label='29.47MB/s', edgecolor='black', linewidth=1.5)
# plt.bar(positions2, b2_SSIM_set, width=bar_width, color='orange', label='9.82MB/s', edgecolor='black', linewidth=1.5)

# plt.bar(positions1, b1_qoe_SSIM_set, width=bar_width, color='skyblue', label='29.47MB/s', edgecolor='black', linewidth=1.5)
# plt.bar(positions2, b2_qoe_SSIM_set, width=bar_width, color='orange', label='9.82MB/s', edgecolor='black', linewidth=1.5)

# plt.bar(positions1, b1_PSNR_set, width=bar_width, color='skyblue', label='29.47MB/s', edgecolor='black', linewidth=1.5)
# plt.bar(positions2, b2_PSNR_set, width=bar_width, color='orange', label='9.82MB/s', edgecolor='black', linewidth=1.5)

plt.bar(positions1, b1_qoe_PSNR_set, width=bar_width, color='skyblue', label='29.47MB/s', edgecolor='black', linewidth=1.5)
plt.bar(positions2, b2_qoe_PSNR_set, width=bar_width, color='orange', label='9.82MB/s', edgecolor='black', linewidth=1.5)

# Add a dashed baseline for avg SSIM
# plt.axhline(y=0.5031, color='black', linestyle='--', linewidth=1.5, label='Baseline (29.47MB/s)')
# plt.axhline(y=0.3600, color='black', linestyle='-.', linewidth=1.5, label='Baseline (9.82MB/s)')
# Add a dashed baseline for qoe SSIM
# plt.axhline(y=0.5844, color='black', linestyle='--', linewidth=1.5, label='Baseline (29.47MB/s)')
# plt.axhline(y=0.3879, color='black', linestyle='-.', linewidth=1.5, label='Baseline (9.82MB/s)')
# Add a dashed baseline for avg PSNR
# plt.axhline(y=20.0445, color='black', linestyle='--', linewidth=1.5, label='Baseline (29.47MB/s)')
# plt.axhline(y=15.5349, color='black', linestyle='-.', linewidth=1.5, label='Baseline (9.82MB/s)')
# Add a dashed baseline for qoe PSNR
plt.axhline(y=28.0425, color='black', linestyle='--', linewidth=1.5, label='Baseline (29.47MB/s)')
plt.axhline(y=22.0618, color='black', linestyle='-.', linewidth=1.5, label='Baseline (9.82MB/s)')

# Add titles and labels
# plt.title('Measurement Result of Average SSIM', fontsize=24)
# plt.title('Measurement Result of QoE(SSIM)', fontsize=24)
# plt.title('Measurement Result of Average PSNR', fontsize=24)
# plt.title('Measurement Result of QoE(PSNR)', fontsize=24)
plt.xlabel('Utility', fontsize=22)
# plt.ylabel('SSIM', fontsize=22)
# plt.ylabel('PSBR(dB)', fontsize=22)
# plt.ylabel('QoE(SSIM)($s^{-1}$)', fontsize=22)
plt.ylabel('QoE(PSBR)(dB/s)', fontsize=22)
plt.xticks([p + bar_width / 2 for p in positions1], categories, fontsize=20)  # Centering category labels
plt.yticks(fontsize=20)
plt.legend(fontsize=16, framealpha=1,loc='lower left')  # Add legend

# Save the plot as an image
# plt.savefig('./result/avg_ssim_bar_plot.png', dpi=300, bbox_inches='tight')  # Adjust DPI and bounding box if needed
# plt.savefig('./result/qoe_ssim_bar_plot.png', dpi=300, bbox_inches='tight')  # Adjust DPI and bounding box if needed
# plt.savefig('./result/avg_psnr_bar_plot.png', dpi=300, bbox_inches='tight')  # Adjust DPI and bounding box if needed
plt.savefig('./result/qoe_psnr_bar_plot.png', dpi=300, bbox_inches='tight')  # Adjust DPI and bounding box if needed

plt.close()  # Close the figure to free memory
