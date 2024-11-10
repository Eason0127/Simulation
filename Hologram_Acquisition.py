import numpy as np
import matplotlib.pyplot as plt

# Setting parameters
wavelengths = np.linspace(531e-9,532e-9, 10)
Z2 = 0.005
Z1 = 0.07
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
    mask = (wavelength * FX) ** 2 + (wavelength * FY) ** 2 > 1
    H[mask] = 0  # Set evanescent wave components to 0
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
aperture_radius = 0.05e-3  # coherence aperture's diameter 1mm
initial_field = np.zeros((resolution, resolution), dtype=complex)  # Initialization
initial_field[(X**2 + Y**2) <= aperture_radius**2] = 1 + 0j
field_at_sample = np.zeros((resolution, resolution), dtype=complex)
for wavelength in wavelengths:
    propagated_field_to_sample = angular_spectrum_approach(initial_field, Z1, wavelength, pixel_size)
    field_at_sample += propagated_field_to_sample

# Define the sample
sample_amplitude = np.ones((resolution, resolution))
sample_phase = np.zeros((resolution, resolution)) # Initialization
sample_radius = 2e-3
sample_phase[(X ** 2 + Y ** 2) <= sample_radius ** 2] = np.pi / 2
sample = sample_amplitude * np.exp(1j * sample_phase)

# Field after the sample
field_after_sample = field_at_sample * sample
total_intensity = np.zeros((resolution, resolution))

# Acquire the hologram
for wavelength in wavelengths:
    propagated_field = angular_spectrum_approach(field_after_sample, Z2, wavelength, pixel_size)
    intensity = np.abs(propagated_field) ** 2
    total_intensity += intensity

# Plot the hologram
plt.figure(figsize=(8, 8))
plt.imshow(total_intensity, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                           resolution // 2 * pixel_size * 1e6,
                                           -resolution // 2 * pixel_size * 1e6,
                                           resolution // 2 * pixel_size * 1e6))
plt.colorbar(label="Intensity")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Hologram at sensor plane")
plt.show()