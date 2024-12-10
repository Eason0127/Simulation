import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def plot_field(field, title="Complex Field", cmap="viridis"):
    # Calculate amplitude and phase
    amplitude = np.abs(field)
    phase = np.angle(field)

    # Normalize phase to range [0, 2π]
    phase = (phase + 2 * np.pi) % (2 * np.pi)

    # Create a figure
    plt.figure(figsize=(12, 6))

    # Plot amplitude
    plt.subplot(1, 2, 1)
    plt.imshow(amplitude, cmap=cmap)
    plt.colorbar(label="Amplitude")
    plt.title(f"{title} - Amplitude")
    plt.axis('off')  # Turn off axis

    # Plot phase
    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap="twilight", vmin=0, vmax=2 * np.pi)
    plt.colorbar(label="Phase (radians)")
    plt.title(f"{title} - Phase")
    plt.axis('off')  # Turn off axis
    # Show plots
    plt.tight_layout()
    plt.show()

def Transfer_function(W, H, distance, wavelength, area):
    FX = W / area
    FY = H / area
    square_root = np.sqrt(1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2))
    temp = np.exp(1j * 2 * np.pi * distance / wavelength * square_root)
    temp[np.isnan(temp)] = 0
    return temp
def angular_spectrum_method(field, area, distance, W, H):
    GT = fftshift(fft2(ifftshift(field)))
    gt_prime = fftshift(ifft2(ifftshift(GT * Transfer_function(W, H, distance, 532e-9, area))))
    return gt_prime


numPixels = 512
pixelSize = 1e-6 # unit: meter
area = numPixels * pixelSize
# Define the sample
Sample_Radius = 50  # pixels * size
Sample_Phase = 3
x = np.arange(numPixels) - numPixels / 2 - 1
y = np.arange(numPixels) - numPixels / 2 - 1
d = x[1] - x[0]
W, H = np.meshgrid(x, y)
print(W, H)

# Define the field after sample
Mask = np.sqrt(W ** 2 + H ** 2) <= Sample_Radius # boundaries of the object
incident_field = np.zeros((numPixels, numPixels), dtype=complex)
incident_field[Mask] = np.exp(1j * Sample_Phase)
plot_field(incident_field)


hologram_field = angular_spectrum_method(incident_field, area, 1e-3, W, H)
hologram_amplitude = np.abs(hologram_field)
plot_field(hologram_field)



# IPR
def IPR(Measured_amplitude, distance, k_max, convergence_threshold, area, W, H):
    update_phase = []
    last_field = None
    rms_errors = []  # Store RMS errors for plotting
    for k in range(k_max):
        # a) sensor plane
        if k == 0:
            phase0= np.zeros(Measured_amplitude.shape)
            field1 = Measured_amplitude * np.exp(1j * phase0)
        else:
            field1 = Measured_amplitude * np.exp(1j * update_phase[k - 1])
        # b) back-propagation and apply energy constraint
        field2 = angular_spectrum_method(field1, area, -distance, W, H)
        phase_field2 = np.angle(field2) # phase
        amp_field2 = np.abs(field2) # amplitude
        abso = -np.log(amp_field2)
        # Apply constraints
        abso[abso < 0] = 0
        phase_field2[abso < 0] = 0
        amp_field2 = np.exp(-abso)
        field22 = amp_field2 * np.exp(1j * phase_field2)
        last_field = field22

        # c) forward propagation and update amplitude
        field3 = angular_spectrum_method(field22, area, distance, W, H)
        amp_field3 = np.abs(field3)
        phase_field3 = np.angle(field3)
        update_phase.append(phase_field3)
        # tell if next iteration is needed
        if k > 0:
            amp_diff = amp_field3 - Measured_amplitude
            rms_error = np.sqrt(np.mean(amp_diff ** 2))
            rms_errors.append(rms_error)
            print(f"the {k} iteration, Error RMS {rms_error}")
            if rms_error < convergence_threshold:  # 小于阈值，认为已收敛
                print(f"Converged at iteration {k}")
                # field_final = Norm_amplitude * np.exp(1j * phase_field3)
                return field22
    # Plot RMS error curve after the iteration ends
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(rms_errors) + 1), rms_errors, marker='o')
    plt.title("RMS Error Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("RMS Error")
    plt.grid()
    plt.show()
    return last_field

# find the image

field_ite = IPR(hologram_amplitude, 3e-4, 1800, 1e-30, area, W, H)
plot_field(field_ite)

