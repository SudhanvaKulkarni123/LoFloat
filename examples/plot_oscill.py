import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load CSV (the file now HAS a header line) ───────────────────────────────
df = pd.read_csv("oscillator_8bit.csv")     # header row auto-detected

# (optional) enforce the expected column order
expected_cols = [
    "t",
    "u_rne_2nd", "v_rne_2nd", "u_sr_2nd", "v_sr_2nd",
    "u_rne_imp", "v_rne_imp", "u_sr_imp", "v_sr_imp",
    "u_rne_mix", "v_rne_mix", "u_sr_mix", "v_sr_mix",
    "u_exact",   "v_exact"
]
df = df[expected_cols]

# ── Force all values to float & clean NaNs/Inf ─────────────────────────────
df = df.apply(pd.to_numeric, errors='coerce')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Extract columns as NumPy arrays ─────────────────────────────────────────
t          = df["t"].to_numpy()

u_rne_2nd  = df["u_rne_2nd"].to_numpy();  v_rne_2nd  = df["v_rne_2nd"].to_numpy()
u_sr_2nd   = df["u_sr_2nd"].to_numpy();   v_sr_2nd   = df["v_sr_2nd"].to_numpy()

u_rne_imp  = df["u_rne_imp"].to_numpy();  v_rne_imp  = df["v_rne_imp"].to_numpy()
u_sr_imp   = df["u_sr_imp"].to_numpy();   v_sr_imp   = df["v_sr_imp"].to_numpy()

u_rne_mix  = df["u_rne_mix"].to_numpy();  v_rne_mix  = df["v_rne_mix"].to_numpy()
u_sr_mix   = df["u_sr_mix"].to_numpy();   v_sr_mix   = df["v_sr_mix"].to_numpy()

u_exact    = df["u_exact"].to_numpy();    v_exact    = df["v_exact"].to_numpy()

# ── Sanity-check for outliers in v(t)  ─────────────────────────────────────
threshold = 10
bad = df[
    (df["v_rne_2nd"].abs() > threshold)  | (df["v_sr_2nd"].abs()  > threshold) |
    (df["v_rne_imp"].abs() > threshold)  | (df["v_sr_imp"].abs()  > threshold) |
    (df["v_rne_mix"].abs() > threshold)  | (df["v_sr_mix"].abs()  > threshold)
]
if not bad.empty:
    print("⚠️  High derivative values detected:\n", bad)

# ── Phase portrait  u(t) vs v(t)  ──────────────────────────────────────────
plt.figure(figsize=(10, 8))

# exact solution
plt.plot(u_exact,   v_exact,   label="Exact", color="black", linewidth=2)

# Heun (explicit midpoint)
plt.plot(u_rne_2nd, v_rne_2nd, label="RNE  Heun 2nd",  linestyle=':')
plt.plot(u_sr_2nd,  v_sr_2nd,  label="SR   Heun 2nd",  linestyle=':')

# implicit midpoint – same precision throughout
plt.plot(u_rne_imp, v_rne_imp, label="RNE  ImpMid",    linestyle='--')
plt.plot(u_sr_imp,  v_sr_imp,  label="SR   ImpMid",    linestyle='--')

# implicit midpoint – mixed precision (float state, 8-bit stages)
plt.plot(u_rne_mix, v_rne_mix, label="RNE  mixed ImpMid", linestyle='-.')
plt.plot(u_sr_mix,  v_sr_mix,  label="SR   mixed ImpMid", linestyle='-.')

plt.xlabel("u(t)")
plt.ylabel("v(t)")
plt.title("Phase Portrait: u(t) vs v(t)")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
