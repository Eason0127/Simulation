import numpy as np
import matplotlib.pyplot as plt

# --- Sampling frequency data (λ = 532 nm, z₂ = 0.5 mm) ---
w = 204.8                # μm
z2_um = 0.5 * 1000       # μm
lambda_um = 0.532        # μm
NA = (w/2) / np.sqrt((w/2)**2 + z2_um**2)
f_system = NA / lambda_um  # μm⁻¹

delta = np.arange(0.2, 3.0, 0.05)  # μm
fs = 1 / (2 * delta)               # μm⁻¹
delta_crit = 1 / (2 * f_system)    # μm

# --- Resolution data for z₂ = 1 mm ---
pitchsize_um = np.arange(0.2, 3.0, 0.05)
resolution_um = np.array([
    5,   5,   5,   5,   5,   5,   5,   5,   5,
    5,   5,   5,   5,   5,   5,   5,   5.5, 5,   5,
    5,   5,   5.5, 6.5, 6,   6.5, 6,   8,   6.5, 8,
    7,   5.5, 5.5, 8,   8,   8.5, 6,   9,   8.5, 9,
    9,   9,   7.5, 7.5,12,  10.5,7.5,10.5,10.5,8.5,
   12,  11,   11,11.5,9.5,14.5,15
])

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot sampling frequency
ax1.plot(delta, fs, color='tab:blue', linewidth=2, label=r'$f_s = 1/(2\Delta)$')
ax1.axhline(f_system, color='tab:red', linestyle='--', linewidth=2,
            label=fr'$f = {f_system:.3f}\,\mu{{m}}^{{-1}}$')
ax1.axvline(delta_crit, color='black', linestyle=':', linewidth=1.5,
            label=fr'Critical $\Delta = {delta_crit:.2f}\,\mu$m')
ax1.scatter([delta_crit], [f_system], color='tab:red', zorder=5)

ax1.set_xlabel('Pitch Size Δ (μm)')
ax1.set_ylabel('Frequency (μm⁻¹)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xlim(0.2, 3.0)
ax1.grid(True, linestyle='--', alpha=0.7)

# Twin axis for resolution
ax2 = ax1.twinx()
ax2.plot(pitchsize_um, resolution_um, color='tab:purple', linewidth=2, label='Resolution (z₂=1 mm)')
ax2.set_ylabel('Reconstructed Resolution (μm)', color='tab:purple')
ax2.tick_params(axis='y', labelcolor='tab:purple')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Sampling Frequency & Resolution vs Pitch Size\n(λ=532 nm)')
plt.tight_layout()
plt.show()
