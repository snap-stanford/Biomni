"""
Generate sample plots for task_002
이미지는 task 폴더 바로 아래에 생성됩니다.
"""
import numpy as np
import matplotlib.pyplot as plt

# 데이터
data = [10, 20, 15, 25, 30, 18, 22, 28, 16, 24]

# 1. 히스토그램
plt.figure(figsize=(8, 6))
plt.hist(data, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of Data', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('histogram.png', dpi=150, bbox_inches='tight')
print("✅ Histogram saved to histogram.png")
plt.close()

# 2. 박스플롯
plt.figure(figsize=(6, 8))
plt.boxplot(data, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightgreen', alpha=0.7),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
plt.ylabel('Value', fontsize=12)
plt.title('Boxplot of Data', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('boxplot.png', dpi=150, bbox_inches='tight')
print("✅ Boxplot saved to boxplot.png")
plt.close()

print("\n📊 All plots generated successfully in current directory!")
print("Run this script from task_002 directory:")
print("  cd qa_datasets/task_002")
print("  python generate_example_plots.py")

