import numpy as np
import matplotlib.pyplot as plt

# Given data for z2 = 1 mm
pitch_size = np.array([
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
    0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05,
    1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55,
    1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05,
    2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55,
    2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0
])

resolution = np.array([
    6, 6, 6, 6, 6, 6, 9, 6, 6, 6,
    6, 6, 8, 6, 6, 6, 6, 6, 6, 5.5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 6, 6, 6, 11, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 7, 7, 7, 8,
    11, 11, 12, 13, 13, 14, 16, 16, 6
])

plt.figure(figsize=(8, 5))
plt.plot(pitch_size, resolution, '-', linewidth=2, color='darkgreen')
plt.xlabel('Pitch Size Δ (μm)')
plt.ylabel('Reconstructed Resolution (μm)')
plt.title('Resolution vs. Pitch Size\n(λ = 532 nm, z₂ = 1 mm)')
plt.xlim(0.1, 3.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
