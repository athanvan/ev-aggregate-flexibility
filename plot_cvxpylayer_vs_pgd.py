import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import argparse

csv_file = "training_times.csv"
out_path = "timing_plot.png"

df = pd.read_csv(csv_file)

df['raw_times'] = df['raw_times'].apply(ast.literal_eval)

def filter_iqr(times):
    series = pd.Series(times).dropna()
    if series.empty:
        return np.nan
    
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    filtered = series[(series >= lower) & (series <= upper)]
    return filtered.tolist()

df['clean_times'] = df['raw_times'].apply(filter_iqr)
df['mean'] = df['clean_times'].apply(np.mean)
df['std'] = df['clean_times'].apply(np.std)


plt.figure(figsize=(8, 5))

for is_cvx, group in df.groupby('cvxpylayers'):
    label = "cvxpylayers" if is_cvx else "PGD"
    color = "C0" if is_cvx else "C1"
    
    group = group.sort_values('N')
    x = group['N']
    y = group['mean']
    yerr = group['std']

    plt.plot(x, y, marker='o', label=label, color=color, linewidth=2)
    
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
    
    plt.plot(x, y + yerr, linestyle='--', color=color, linewidth=1, alpha=0.5)
    plt.plot(x, y - yerr, linestyle='--', color=color, linewidth=1, alpha=0.5)

plt.xlabel("$N$ and $T$")
plt.ylabel("time per epoch (sec)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(out_path, dpi=200)
print(f"Plot saved to {out_path}")
plt.show()