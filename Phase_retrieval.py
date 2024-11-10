import numpy as np
import matplotlib.pyplot as plt

# Setting parameters
wavelengths = np.linspace(531e-9,532e-9, 10)
Z2 = 0.005
Z1 = 0.07
resolution = 1024
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
    H[mask] = 0
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
sample_radius = 1e-4
sample_phase[(X ** 2 + Y ** 2) <= sample_radius ** 2] = np.pi / 2
sample = sample_amplitude * np.exp(1j * sample_phase)

# Field after the sample
field_after_sample = field_at_sample * sample
total_intensity_at_sample = np.zeros((resolution, resolution))
field_at_sensor = np.zeros((resolution, resolution), dtype=complex) # Initialization

# Acquire the hologram
for wavelength in wavelengths:
    propagated_field_to_sensor = angular_spectrum_approach(field_after_sample, Z2, wavelength, pixel_size)
    field_at_sensor += propagated_field_to_sensor

total_intensity_at_sample = np.abs(field_at_sensor) ** 2

# Plot the hologram
plt.figure(figsize=(8, 8))
plt.imshow(total_intensity_at_sample, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                           resolution // 2 * pixel_size * 1e6,
                                           -resolution // 2 * pixel_size * 1e6,
                                           resolution // 2 * pixel_size * 1e6))
plt.colorbar(label="Intensity")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Hologram at sensor plane")
plt.show()

# Apply back-propagation to this hologram to make an initial guess
total_intensity_of_initial_guess = np.zeros((resolution, resolution))
field_of_initial_guess = np.zeros((resolution, resolution), dtype=complex) # Initialization
for wavelength in wavelengths:
    back_propagated_field_to_sample = angular_spectrum_approach(field_at_sensor, -Z2, wavelength, pixel_size)
    field_of_initial_guess += back_propagated_field_to_sample

phase = np.angle(field_of_initial_guess)
total_intensity_of_initial_guess = np.abs(field_of_initial_guess) ** 2

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(total_intensity_of_initial_guess, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                                resolution // 2 * pixel_size * 1e6,
                                                -resolution // 2 * pixel_size * 1e6,
                                                resolution // 2 * pixel_size * 1e6))
plt.colorbar(label="Amplitude")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Initial guess of Amplitude at Sample Plane")

plt.subplot(1, 2, 2)
plt.imshow(phase, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                            resolution // 2 * pixel_size * 1e6,
                                            -resolution // 2 * pixel_size * 1e6,
                                            resolution // 2 * pixel_size * 1e6))
plt.colorbar(label="Phase")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Initial guess of Phase at Sample Plane")

plt.show()



