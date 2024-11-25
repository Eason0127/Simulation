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
    fx = np.fft.fftfreq(resolution, d=pixel_size) # check
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

def propagation(initial_field, distance):
    propagated_intensity = np.zeros((resolution, resolution))
    field = np.zeros((resolution, resolution), dtype=complex)
    for wavelength in wavelengths:
        propagated_field = angular_spectrum_approach(initial_field, distance, wavelength, pixel_size)
        propagated_intensity += np.abs(propagated_field) ** 2  # Cumulative Strength
        field += propagated_field  # Accumulate complex fields (for phase calculation)
    # Calculate the average phase
    phase = np.angle(field)
    intensity = propagated_intensity
    return intensity, phase, field

def plot(field, phase, title_1, title_2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(field, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                                         resolution // 2 * pixel_size * 1e6,
                                                         -resolution // 2 * pixel_size * 1e6,
                                                         resolution // 2 * pixel_size * 1e6))
    plt.colorbar(label="Amplitude")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='gray', extent=(-resolution // 2 * pixel_size * 1e6,
                                                     resolution // 2 * pixel_size * 1e6,
                                                     -resolution // 2 * pixel_size * 1e6,
                                                     resolution // 2 * pixel_size * 1e6))
    plt.colorbar(label="Phase")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(title_2)
    plt.show()

# Sensor plane coordination creation
x = np.linspace(-resolution // 2, resolution // 2 - 1, resolution) * pixel_size
y = np.linspace(-resolution // 2, resolution // 2 - 1, resolution) * pixel_size
X, Y = np.meshgrid(x, y)

# Define the incident light
aperture_radius = 0.05e-3  # coherence aperture's diameter 1mm
initial_field = np.zeros((resolution, resolution), dtype=complex)  # Initialization
initial_field[(X**2 + Y**2) <= aperture_radius**2] = 1 + 0j
intensity_at_sample, phase_at_sample, field_at_sample = propagation(initial_field,Z1)

# Define the sample
sample_amplitude = np.ones((resolution, resolution)) # Initialization
sample_phase_delay = np.zeros((resolution, resolution)) # Initialization
sample_radius = 1e-4
sample_phase_delay[(X ** 2 + Y ** 2) <= sample_radius ** 2] = np.pi / 2
sample = sample_amplitude * np.exp(1j * sample_phase_delay)

# Field after the sample
field_after_sample = field_at_sample * sample

# Acquire the hologram
intensity_of_origin_hologram, phase_of_origin_hologram, field_at_sensor = propagation(field_after_sample, Z2)

# Plot the hologram
plot(intensity_of_origin_hologram, phase_of_origin_hologram, "Intensity of origin hologram", "Phase of origin hologram")

# Apply back-propagation to this hologram to make an initial guess
intensity_of_initial_guess, phase_of_initial_guess, field_of_initial_guess = propagation(field_at_sensor, -Z2)

# Plot the initial guess
plot(intensity_of_initial_guess, phase_of_initial_guess, "Intensity of initial guess", "Phase of initial guess")

# Forward-propagate the updated field to the sensor plane
new_field_intensity_at_senor, new_field_phase_at_sensor, new_field_at_sensor_plane = propagation(field_of_initial_guess, Z2)

# Update the amplitude with a weighted average:60% of the newly forward-propagated field and,40% of the measured one
amplitude_of_hologram = np.sqrt(intensity_of_origin_hologram)
amplitude_of_new_field = np.abs(new_field_at_sensor_plane)
updated_amplitude = 0.6 * amplitude_of_new_field + 0.4 * amplitude_of_hologram
updated_field = updated_amplitude * np.exp(1j * new_field_phase_at_sensor)

# Plot the new field
# plot(new_field_intensity_at_senor, new_field_phase_at_sensor, "Intensity of the new hologram", "Phase of the new hologram")

# back-propagate to the sample plane again
new_field_intensity_at_sample, new_field_phase_at_sample, new_field_at_sample_plane = propagation(updated_field, -Z2)


# Plot the new field at sample plane
plot(new_field_intensity_at_sample, new_field_phase_at_sample, "Intensity of the new field", "Phase of the new field")
