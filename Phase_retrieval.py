import numpy as np
import matplotlib.pyplot as plt

# Setting parameters
wavelengths = np.linspace(528e-9,532e-9, 10)
Z2 = 0.005
resolution = 512
pixel_size = 1.3e-6

# Definition of the angular spectrum approach
def angular_spectrum_approach(complex_field, distance, wavelength, pixel_size):
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(resolution, d=pixel_size)
    fy = np.fft.fftfreq(resolution, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    # Transfer function
    H = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
    # Propagation
    A = np.fft.fft2(complex_field)  # Angular spectrum
    AZ = A * H  # Apply the transfer function
    propagated_field = np.fft.ifft2(AZ)  # inverse furier transform
    return propagated_field

# Sensor plane coordination creation
x = np.linspace(-resolution // 2, resolution // 2 - 1, resolution) * pixel_size
y = np.linspace(-resolution // 2, resolution // 2 - 1, resolution) * pixel_size
X, Y = np.meshgrid(x, y)

# Define the incident light
initial_field = np.ones((resolution, resolution), dtype=complex)

# Define the sample
sample_amplitude = np.ones((resolution, resolution))
sample_phase = np.zeros((resolution, resolution)) # Initialization
sample_radius = 0.5e-3
sample_phase[(X ** 2 + Y ** 2) <= sample_radius ** 2] = np.pi / 2
sample = sample_amplitude * np.exp(1j * sample_phase)

# Field after the sample
field_after_sample = initial_field * sample
total_intensity = np.zeros((resolution, resolution))

# Acquire the hologram
for wavelength in wavelengths:
    propagated_field = angular_spectrum_approach(field_after_sample, Z2, wavelength, pixel_size)
    intensity = np.abs(propagated_field) ** 2
    total_intensity += intensity

# Plot the hologram
plt.figure(figsize=(6, 6))
plt.imshow(total_intensity, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                           resolution // 2 * pixel_size * 1e6,
                                           -resolution // 2 * pixel_size * 1e6,
                                           resolution // 2 * pixel_size * 1e6))
plt.colorbar(label="Intensity")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Hologram at the sensor")
plt.show()