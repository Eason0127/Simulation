import numpy as np
import matplotlib.pyplot as plt

# 设置参数
wavelengths = np.linspace(528e-9,534e-9, 10) # Wavelength along 530nm with 4nm bandwidth
distance = 0.07  # Propagation distance
resolution = 512  # Resolution
pixel_size = 1.3e-6  # Pixel size(1.3 µm）

# Definition of angular spectrum approach
def angular_spectrum_approach(complex_field, distance, wavelength, pixel_size):
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(resolution, d=pixel_size)  # GPT said do it like this, i don't know QAQ
    fy = np.fft.fftfreq(resolution, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    # transfer function
    H = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
    A = np.fft.fft2(complex_field)  # Angular spectrum
    # Apply transfer function to propagation
    AZ = A * H  # 应用传播函数
    propagated_field = np.fft.ifft2(AZ)  # inverse fourier transform
    return propagated_field

# Create coordination at sensor plane
x = np.linspace(-resolution // 2, resolution // 2 - 1, resolution) * pixel_size
y = np.linspace(-resolution // 2, resolution // 2 - 1, resolution) * pixel_size
X, Y = np.meshgrid(x, y)

# Define the aperture
aperture_radius = 0.05e-3
initial_field = np.zeros((resolution, resolution), dtype=complex)  # Initialization
initial_field[(X**2 + Y**2) <= aperture_radius**2] = 1 + 0j

# Propagation
total_intensity = np.zeros((resolution, resolution))

for wavelength in wavelengths:
    propagated_field = angular_spectrum_approach(initial_field, distance, wavelength, pixel_size)
    intensity = np.abs(propagated_field) ** 2
    total_intensity += intensity

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(total_intensity, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                                 resolution // 2 * pixel_size * 1e6,
                                                 -resolution // 2 * pixel_size * 1e6,
                                                 resolution // 2 * pixel_size * 1e6))
plt.colorbar(label="Intensity")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Diffraction Pattern at 7 cm with 4 nm Bandwidth Light Source")
plt.show()
