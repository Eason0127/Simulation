import numpy as np
import matplotlib.pyplot as plt

# Setting parameters
wavelengths = np.array([623e-9, 523e-9, 460e-9])
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

def propagation(initial_field, distance):
    propagated_intensity = np.zeros((resolution, resolution))
    propagated_phase = np.zeros((resolution, resolution), dtype=complex)
    for wavelength in wavelengths:
        propagated_field = angular_spectrum_approach(initial_field, distance, wavelength, pixel_size)
        propagated_intensity += np.abs(propagated_field) ** 2  # Cumulative Strength
        propagated_phase += propagated_field  # Accumulate complex fields (for phase calculation)
    # Calculate the average phase
    phase = np.angle(propagated_phase)
    intensity = propagated_intensity
    return intensity, phase, propagated_phase

def one_wavelength_propagation(field, wavelength, distance):
    propagated_field = angular_spectrum_approach(field, distance, wavelength, pixel_size)
    return propagated_field

def amplitude_update(amplitude_recorded, amplitude_of_initial_guess, phase_of_initial_guess, distance, number_of_wavelengths):
        sum = np.zeros((resolution, resolution), dtype=complex)
        for wavelength in wavelengths:
            phase = wavelengths[0] / wavelength * phase_of_initial_guess
            field = amplitude_of_initial_guess * np.exp(1j * phase)
            FP_field = one_wavelength_propagation(field, wavelength, distance)  # forward propagate to the sensor
            phase_of_FP_field = np.angle(FP_field)
            FP_field_new = amplitude_recorded * np.exp(1j * phase_of_FP_field)
            BP_field = one_wavelength_propagation(FP_field_new, wavelength, -distance)
            BP_field_amplitude = np.abs(BP_field)
            BP_field_phase = np.angle(BP_field)
            # Energy constrains
            mask = BP_field_amplitude > 1
            BP_field[mask] = np.exp(1j * BP_field_phase[mask])
            sum += BP_field
        averaged_field = sum / number_of_wavelengths
        return averaged_field


def update_and_propagation(intensity_norm, initial_guess_field, distance, number_of_wavelengths, k_max, loss):
    next_amplitude = []
    initial_phase = []
    updated_phase = []
    g_k = []
    alpha = 0  # initialize alpha
    amplitude_of_hologram = np.sqrt(intensity_norm)
    phase_of_initial_guess = np.angle(initial_guess_field)
    initial_phase.append(phase_of_initial_guess)
    next_amplitude.append(amplitude_of_hologram)

    for k in range(k_max):
        # upgrade the amplitude
        averaged_field = amplitude_update(next_amplitude[k], next_amplitude[k], initial_phase[k], distance, number_of_wavelengths)
        averaged_field_amplitude = np.abs(averaged_field)
        averaged_field_phase = np.angle(averaged_field)
        # upgrade the arrays
        updated_phase.append(averaged_field_phase)
        next_amplitude.append(averaged_field_amplitude)
        if k == 0:
            gk = updated_phase[k] - initial_phase[k]
            g_k.append(gk)
            phase_new = initial_phase[k]
            initial_phase.append(phase_new)
        else:
            h_k = initial_phase[k] - initial_phase[k - 1]
            gk = updated_phase[k] - initial_phase[k]
            g_k.append(gk)
            # 计算 alpha
            alpha = 0  # 初始化 alpha
            for i in range(k, 1, -1):
                n = (g_k[i] * g_k[i - 1]) / (g_k[i - 1] * g_k[i - 1])
                alpha += n
            # update the phase
            phase_new = initial_phase[k] + alpha * h_k
            initial_phase.append(phase_new)
        # check the loss
        norm = np.sqrt(np.linalg.norm(next_amplitude[k + 1] - np.sqrt(intensity_norm)))
        print(f"Iteration {k}: norm = {norm}, loss = {loss}")
        if norm < loss:
            return averaged_field
    return next_amplitude[-1] * np.exp(1j * initial_phase[-1])

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
intensity_background, phase_background, field_background = propagation(initial_field, Z1 + Z2)
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
# plot(intensity_of_origin_hologram, phase_of_origin_hologram, "Intensity of origin hologram", "Phase of origin hologram")

# Apply back-propagation to this hologram to make an initial guess
intensity_of_initial_guess, phase_of_initial_guess, field_of_initial_guess = propagation(field_at_sensor, -Z2)

# Plot the initial guess
# plot(intensity_of_initial_guess, phase_of_initial_guess, "Intensity of initial guess", "Phase of initial guess")
intensity_norm = intensity_at_sample / intensity_background
new_field = update_and_propagation(field_at_sensor, intensity_norm, field_of_initial_guess, Z2, 3, 20, 0.001)