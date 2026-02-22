'''
In The name of GOD

Author : Ali Pilehvar Meibody

Gather all model metrics and plot them in a single figure.
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

# =====================================================================
# All metrics collected from tuned cross-validation results
# Structure:  metrics[model][target] = {metric: (mean, std)}
# =====================================================================

metrics = {
    'LR': {
        'L*': {'RMSE': (5.1868, 2.1298), 'MAE': (3.5022, 1.2189), 'R²': (-0.1697, 0.4722), 'MAPE': (0.1624, 0.0729)},
        'a*': {'RMSE': (4.6418, 1.8340), 'MAE': (3.6635, 1.5506), 'R²': (0.3576, 0.9755), 'MAPE': (0.1314, 0.0679)},
        'b*': {'RMSE': (5.1545, 1.8387), 'MAE': (4.0769, 1.5502), 'R²': (0.3035, 1.4191), 'MAPE': (0.2123, 0.1188)},
    },
    'KNN': {
        'L*': {'RMSE': (4.6741, 2.0658), 'MAE': (3.0049, 0.9319), 'R²': (0.0501, 0.3111), 'MAPE': (0.1348, 0.0536)},
        'a*': {'RMSE': (4.2471, 1.8019), 'MAE': (3.2338, 1.4590), 'R²': (0.4622, 0.5806), 'MAPE': (0.1153, 0.0801)},
        'b*': {'RMSE': (4.5235, 1.5424), 'MAE': (3.4524, 1.0496), 'R²': (0.4636, 0.6509), 'MAPE': (0.1862, 0.1226)},
    },
    'DT': {
        'L*': {'RMSE': (5.2711, 2.6536), 'MAE': (3.4920, 1.4551), 'R²': (-0.2081, 0.5856), 'MAPE': (0.1268, 0.0752)},
        'a*': {'RMSE': (4.7529, 1.9101), 'MAE': (3.5481, 1.7399), 'R²': (0.3264, 0.5568), 'MAPE': (0.1029, 0.0943)},
        'b*': {'RMSE': (6.0071, 1.6037), 'MAE': (4.1091, 1.1643), 'R²': (0.0540, 0.6783), 'MAPE': (0.1594, 0.1263)},
    },
    'RF': {
        'L*': {'RMSE': (4.8656, 2.3351), 'MAE': (3.0191, 1.2136), 'R²': (-0.0294, 0.4527), 'MAPE': (0.1355, 0.0608)},
        'a*': {'RMSE': (4.1331, 1.6789), 'MAE': (3.1353, 1.3536), 'R²': (0.4906, 0.6083), 'MAPE': (0.1169, 0.0823)},
        'b*': {'RMSE': (4.5864, 1.6683), 'MAE': (3.4567, 1.2070), 'R²': (0.4486, 0.5761), 'MAPE': (0.1906, 0.1301)},
    },
    'SVR': {
        'L*': {'RMSE': (3.1376, 0.7809), 'MAE': (2.7194, 0.7120), 'R²': (0.5720, 0.6042), 'MAPE': (0.1240, 0.0404)},
        'a*': {'RMSE': (3.6258, 0.9991), 'MAE': (3.0036, 0.8878), 'R²': (0.6080, 0.2346), 'MAPE': (0.1102, 0.0439)},
        'b*': {'RMSE': (3.4893, 1.1174), 'MAE': (2.6521, 0.9072), 'R²': (0.6808, 0.2609), 'MAPE': (0.1429, 0.0633)},
    },
    'MLP': {
        'L*': {'RMSE': (4.0597, 1.7215), 'MAE': (2.6808, 0.8296), 'R²': (0.2834, 0.3643), 'MAPE': (0.1136, 0.0363)},
        'a*': {'RMSE': (4.3552, 1.0106), 'MAE': (3.5847, 0.8659), 'R²': (0.4345, 1.0211), 'MAPE': (0.1295, 0.0535)},
        'b*': {'RMSE': (3.9104, 1.2828), 'MAE': (2.9813, 0.9373), 'R²': (0.5992, 0.4099), 'MAPE': (0.1685, 0.0969)},
    },
}

# =====================================================================
# Plot: 2x2 grid — one subplot per metric, grouped bars by target
# =====================================================================

models = list(metrics.keys())
targets = ['L*', 'a*', 'b*']
metric_names = ['RMSE', 'MAE', 'R²', 'MAPE']

target_colors = {
    'L*': '#2196F3',
    'a*': '#E53935',
    'b*': '#43A047',
}

n_models = len(models)
n_targets = len(targets)
bar_width = 0.22
x = np.arange(n_models)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cross-Validated Model Performance Comparison', fontsize=16, fontweight='bold', y=0.98)

for ax, metric in zip(axes.flat, metric_names):
    for j, target in enumerate(targets):
        means = [metrics[m][target][metric][0] for m in models]
        stds  = [metrics[m][target][metric][1] for m in models]
        offset = (j - 1) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width,
            yerr=stds,
            label=target,
            color=target_colors[target],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.6,
            capsize=3,
            error_kw={'elinewidth': 1, 'capthick': 1, 'alpha': 0.7},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_title(metric, fontsize=13, fontweight='bold', pad=8)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    if metric == 'R²':
        ax.axhline(y=0, color='grey', linewidth=0.8, linestyle='-', alpha=0.5)

    if metric == 'MAPE':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

axes[0, 0].legend(title='Target', fontsize=9, title_fontsize=10, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print('Plot saved to metrics_comparison.png')

# =====================================================================
# Export metrics table to CSV
# =====================================================================

csv_path = 'metrics_table.csv'

header = ['Model', 'Target',
          'RMSE_mean', 'RMSE_std',
          'MAE_mean',  'MAE_std',
          'R2_mean',   'R2_std',
          'MAPE_mean', 'MAPE_std']

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for model in models:
        for target in targets:
            row = [model, target]
            for metric in metric_names:
                mean, std = metrics[model][target][metric]
                row.extend([f'{mean:.4f}', f'{std:.4f}'])
            writer.writerow(row)

print(f'CSV saved to {csv_path}')
