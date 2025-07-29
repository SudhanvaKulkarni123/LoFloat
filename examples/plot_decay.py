import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("stiff_decay_8bit.csv")

# Plot settings
plt.figure(figsize=(10, 6))
plt.plot(df["t"], df["y_exact"], label="Exact Solution $e^{-15t}$", linewidth=2)
plt.plot(df["t"], df["y_rne"], 'o-', label="RNE (8-bit Euler)", markersize=4)
plt.plot(df["t"], df["y_sr"], 's--', label="SR (8-bit Euler)", markersize=4)

plt.title("Stiff ODE: $dy/dt = -15y$ using Low-Precision Euler", fontsize=14)
plt.xlabel("Time $t$", fontsize=12)
plt.ylabel("$y(t)$", fontsize=12)
plt.yscale('log')  # optional: use log-scale to show tiny values
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("stiff_decay_plot.png", dpi=300)
plt.show()
