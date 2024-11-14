import numpy as np
import matplotlib.pyplot as plt

# Setting parameters
wavelengths = np.linspace(531e-9,532e-9, 10)
Z2 = 0.005
Z1 = 0.07
resolution = 1024
pixel_size = 1.3e-6
k_max = 100 # 100 iterations
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
    field = np.zeros((resolution, resolution), dtype=complex)
    for wavelength in wavelengths:
        propagated_field = angular_spectrum_approach(initial_field, distance, wavelength, pixel_size)
        field += propagated_field # !!!!! Have problem. Remain to solve
    return field

def one_wavelength_propagation(field, wavelength, distance):
    propagated_field = angular_spectrum_approach(field, distance, wavelength, pixel_size)
    return propagated_field
def update_and_propagation(field_hologram, initial_guess_field, distance, number_of_wavelengths, k_max):
    next_amplitude = [] # amplitude of the newest field
    origin_phase = [] # phase before propagation
    phase_for_compute = [] # phase of the newest field
    gk = [] # g_k

    for k in range(k_max):
        if k == 0:
            amplitude_of_hologram = np.abs(field_hologram)
            phase_of_initial_guess = np.angle(initial_guess_field)
            next_amplitude.append(phase_of_initial_guess) # Put initial guess in the related array
            sum = np.zeros((resolution, resolution), dtype=complex)
            for wavelength in wavelengths:
                phase = wavelengths[0] / wavelength * phase_of_initial_guess
                field = amplitude_of_hologram * np.exp(1j * phase)
                FP_field = one_wavelength_propagation(field, wavelength, distance)  # forward propagate to the sensor
                BP_field = one_wavelength_propagation(FP_field, wavelength, -distance)
                BP_field_amplitude = np.abs(BP_field)
                BP_field_phase = np.angle(BP_field)
                # Energy constrains
                mask = BP_field_amplitude > 1
                BP_field[mask] = np.exp(1j * BP_field_phase)
                sum += BP_field
            # Global update
            averaged_field = sum / number_of_wavelengths
            averaged_field_amplitude = np.abs(averaged_field)
            averaged_field_phase = np.angle(averaged_field)
            # Put the data in the related array for k = 0
            phase_for_compute.append(averaged_field_phase)
            gk.append(0)
            # Put the data in the related array for k = 1
            next_amplitude.append(averaged_field_amplitude)
            origin_phase.append(averaged_field_phase)
        else:
            h_k = origin_phase[k] - origin_phase[k - 1]
            g_k = phase_for_compute[k] - origin_phase[k]
            gk.append(g_k)
            for i in range(k, 1, -1):
                n = (gk[i] * gk[i - 1]) / (gk[i - 1] * gk[i - 1])
                alpha_k += n
            phase_k_next = origin_phase[k] + alpha_k * h_k
            origin_phase.append(phase_k_next)
            field_k = next_amplitude[k - 1] * np.exp(1j * origin_phase[k])
            field_k_next = next_amplitude[k] *






def MWPREGV(field, distance):

    # back-propagate to the sample plane again
    new_field_at_sample_plane = propagation(new_field_at_sensor_plane, -distance)
    new_field_phase_at_sample = np.angle(new_field_at_sample_plane)
    new_field_amplitude_at_sample = np.abs(new_field_at_sample_plane)
    mask = new_field_amplitude_at_sample > 1
    new_field_at_sample_plane[mask] = np.exp(1j * new_field_phase_at_sample) # Eliminating the twin image

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
field_at_sample = propagation(initial_field,Z1)

# Define the sample
sample_amplitude = np.ones((resolution, resolution)) # Initialization
sample_phase_delay = np.zeros((resolution, resolution)) # Initialization
sample_radius = 1e-4
sample_phase_delay[(X ** 2 + Y ** 2) <= sample_radius ** 2] = np.pi / 2
sample = sample_amplitude * np.exp(1j * sample_phase_delay)

# Field after the sample
field_after_sample = field_at_sample * sample

# Acquire the hologram
field_at_sensor = propagation(field_after_sample, Z2)
intensity_of_origin_hologram = np.abs(field_at_sensor) ** 2
phase_of_origin_hologram = np.angle(field_at_sensor)

# Apply back-propagation to all the holograms and make an average (but here we just have one hologram)
field_of_initial_guess = propagation(field_at_sensor, -Z2)