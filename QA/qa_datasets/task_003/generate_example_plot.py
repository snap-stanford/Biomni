"""
Generate example fold change plot for task_003
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('gene_expression_data.csv')

# 평균 계산
df['treatment_mean'] = df[['sample1', 'sample2', 'sample3']].mean(axis=1)
df['control_mean'] = df[['control1', 'control2', 'control3']].mean(axis=1)

# Fold change 계산
df['fold_change'] = df['treatment_mean'] / df['control_mean']
df['log2_fold_change'] = np.log2(df['fold_change'])

# DEG 판별 (간단히)
df['regulation'] = 'NS'
df.loc[(df['log2_fold_change'] > 1), 'regulation'] = 'Up'
df.loc[(df['log2_fold_change'] < -1), 'regulation'] = 'Down'

# 시각화
plt.figure(figsize=(10, 6))
colors = ['red' if x == 'Up' else 'blue' if x == 'Down' else 'gray' 
          for x in df['regulation']]

plt.bar(df['gene_id'], df['log2_fold_change'], color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Log2FC = 1')
plt.axhline(y=-1, color='blue', linestyle='--', linewidth=1, label='Log2FC = -1')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.xlabel('Gene', fontsize=12, fontweight='bold')
plt.ylabel('Log2 Fold Change', fontsize=12, fontweight='bold')
plt.title('Differential Gene Expression Analysis', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fold_change_plot.png', dpi=300)
print("✅ Plot saved: fold_change_plot.png")

print("\nLog2 Fold Change values:")
print(df[['gene_id', 'log2_fold_change', 'regulation']])

