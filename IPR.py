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

def Transfer_function(FX, FY, z, wavelength):
    square_root = np.sqrt(1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2))
    temp = np.exp(1j * 2 * np.pi * z / wavelength * square_root)
    temp[np.isnan(temp)] = 0  # replace nan's with zeros
    return temp
def angular_spectrum_method(field, sampling_distance, distance, FX, FY):
    GT = ifftshift(fft2(fftshift(field))) * sampling_distance ** 2
    # * dx ** 2 :Make the discrete Fourier transform result numerically close to the continuous Fourier transform result and Maintain power consistency
    gt_prime = fftshift(ifft2(ifftshift(GT * Transfer_function(FX, FY, distance, 532e-9)))) / sampling_distance ** 2
    return gt_prime

numPixels = 512
pixelSize = 1e-7 # microns
# Define the sample
Sample_Radius = 50  # In pixels
Sample_Phase = 0.75
Center = numPixels / 2 # assumes numPixels is even
W, H = np.meshgrid(np.arange(0, numPixels), np.arange(0, numPixels)) # coordinates of the array indexes


# Define the field after sample
Mask = np.sqrt((W - Center)**2 + (H - Center)**2) <= Sample_Radius # boundaries of the object
exit_field = np.ones((numPixels, numPixels), dtype=complex)
exit_field[Mask] = np.exp(-0.1) * np.exp(1j * Sample_Phase)
# plot_field(exit_field)

x = np.linspace(-pixelSize * numPixels / 2, pixelSize * numPixels / 2, num = numPixels, endpoint = True)
physicalRadius = Sample_Radius * pixelSize
dx = x[1] - x[0]    # Sampling period
fS = 1 / dx         # Spatial sampling frequency
df = fS / numPixels # Spacing between discrete frequency coordinates
fx = np.arange(-fS / 2, fS / 2, step = df) # Spatial frequency, inverse microns
FX, FY = np.meshgrid(fx, fx)

hologram_field = angular_spectrum_method(exit_field, dx, 1e-4, FX, FY)


# plot_field(hologram_field)

# fig, ax = plt.subplots(nrows=1, ncols=1)
# img = ax.imshow(np.abs(hologram_field)**2)
# img.set_extent([x[0], x[-1], x[0], x[-1]])
# cb = plt.colorbar(img)
# cb.set_label('amplitude$^2$')
# ax.set_xticks([x[0], x[-1]])
# ax.set_yticks([x[0], x[-1]])
# ax.set_xlabel(r'$x, \, \mu m$')
# ax.set_ylabel(r'$y, \, \mu m$')
# plt.show()

# Get the background intensity
background_field = np.ones((numPixels, numPixels), dtype=complex)
Field_no_sample = angular_spectrum_method(background_field, dx, 1e-4, FX, FY)
Intensity_background = np.abs(Field_no_sample) ** 2
print(Intensity_background)

# Normalization
hologram_intensity = np.abs(hologram_field) ** 2
print(hologram_intensity)
Norm_amplitude = np.sqrt(hologram_intensity / Intensity_background)

# IPR
def IPR(Norm_amplitude, distance, k_max, convergence_threshold, support, FX, FY):
    update_phase = []
    for k in range(k_max):
        # a) sensor plane
        if k == 0:
            phase0= 0
            field1 = Norm_amplitude * np.exp(1j * phase0)
        else:
            field1 = Norm_amplitude * np.exp(1j * update_phase[k - 1])
        # b) back-propagation and apply energy constraint
        field2 = angular_spectrum_method(field1, dx, -distance, FX, FY)
        field2 = field2 * support
        phase_field2 = np.angle(field2)
        amp_field2 = np.abs(field2)
        mask = amp_field2 > 1
        field2[mask] = np.exp(1j * phase_field2[mask])
        # c) forward propagation and update amplitude
        field3 = angular_spectrum_method(field2, dx, distance, FX, FY)
        phase_field3 = np.angle(field3)
        update_phase.append(phase_field3)
        # tell if next iteration is needed
        if k > 0:  # 从第 1 次迭代开始比较相位差
            phase_diff = np.abs(update_phase[k] - update_phase[k - 1])  # 计算相位差
            max_diff = np.mean(phase_diff)  # 或 np.mean(phase_diff) 查看全局变化
            print(f"the {k} iteration, max diff {max_diff}")
            if max_diff < convergence_threshold:  # 如果相位变化小于阈值，认为已收敛
                print(f"Converged at iteration {k}")
                field_final = Norm_amplitude * np.exp(1j * phase_field3)
                return field_final

# find the image
support_mask = np.zeros_like(Norm_amplitude)
support_mask[Mask] = 1 * np.exp(-0.1)
field_ite = IPR(Norm_amplitude, 1e-4, 1000, 1e-4, support_mask, FX, FY)
plot_field(field_ite)
image_field = angular_spectrum_method(field_ite, dx, -1e-4, FX, FY)
plot_field(image_field)



