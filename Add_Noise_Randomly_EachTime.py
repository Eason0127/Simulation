import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from scipy.ndimage import gaussian_filter

# Obey the Shannon criteria and with highpass/lowpass filter
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


def load_image(filepath, normalize=True, to_grayscale=True):
    # Load the image
    image = Image.open(filepath)
    # Convert to grayscale if required
    if to_grayscale:
        image = image.convert('L')  # 'L' mode is 8-bit grayscale
    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)
    # Normalize the pixel values to the range [0, 1] if required
    if normalize:
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    return image_array
def apply_gaussian_window(field, W, H, area, sigma=1.0):
    # Compute Gaussian weights
    FX = W / area
    FY = H / area
    gaussian_window = np.exp(-(FX**2 + FY**2) / (2 * sigma**2))
    return field * gaussian_window

def band_pass_filter(W, H, area, low_cutoff=None, high_cutoff=None):
    FX = W / area
    FY = H / area
    spatial_frequencies = np.sqrt(FX**2 + FY**2)  # Radial spatial frequencies
    # Initialize mask
    band_pass_mask = np.ones_like(spatial_frequencies, dtype=bool)
    # Apply high-pass filter if specified
    if low_cutoff is not None:
        high_pass_mask = spatial_frequencies >= low_cutoff
        band_pass_mask &= high_pass_mask
    # Apply low-pass filter if specified
    if high_cutoff is not None:
        low_pass_mask = spatial_frequencies <= high_cutoff
        band_pass_mask &= low_pass_mask
    return band_pass_mask



def Transfer_function(W, H, distance, wavelength, area):
    # Calculate spatial frequencies
    FX = W / area
    FY = H / area
    # Calculate the square root for the transfer function
    square_root = np.sqrt(1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2))
    valid_mask = (wavelength ** 2 * FX ** 2 + wavelength ** 2 * FY ** 2) <= 1
    square_root[~valid_mask] = 0
    # Apply the transfer function formula
    temp = np.exp(1j * 2 * np.pi * distance / wavelength * square_root)
    return temp


def angular_spectrum_method(field, area, distance, W, H, wavelength, low_cutoff=None, high_cutoff=None, sigma=None):
    # Perform Fourier Transform
    GT = fftshift(fft2(ifftshift(field)))
    # Compute transfer function
    transfer_function = Transfer_function(W, H, distance, wavelength, area)
    # Apply transfer function
    GT_filtered = GT * transfer_function
    # Apply band-pass filter
    band_pass_mask = band_pass_filter(W, H, area, low_cutoff, high_cutoff)
    GT_filtered = GT_filtered * band_pass_mask
    # Optionally apply Gaussian window in frequency domain
    if sigma is not None:
        GT_filtered = apply_gaussian_window(GT_filtered, W, H, area, sigma=sigma)
    # Perform Inverse Fourier Transform
    gt_prime = fftshift(ifft2(ifftshift(GT_filtered)))
    return gt_prime

wavelength = 532e-9
numPixels = 500
pixelSize = 4e-6 # unit: meter
area = numPixels * pixelSize
z = 0.05
max_frq = 1 / 532e-9
min_frq = 0

# Coordination of sensor
x = np.arange(numPixels) - numPixels / 2 - 1
y = np.arange(numPixels) - numPixels / 2 - 1
W, H = np.meshgrid(x, y)

# hologram
hologram_field = load_image('pic/b_hologram.jpg')
print(hologram_field)
hologram_amplitude = np.sqrt(hologram_field)
print(hologram_amplitude)
plot_field(hologram_amplitude)



# IPR
# def IPR(Measured_amplitude, distance, wavelength, k_max, convergence_threshold, area, W, H, min_frq, max_frq):
#     update_phase = []
#     last_field = None
#     rms_errors = []  # Store RMS errors for plotting
#     noise_iterations = {250}  # Iterations to add random noise
#     for k in range(k_max):
#         # a) sensor plane
#         if k == 0:
#             phase0 = np.zeros(Measured_amplitude.shape)
#             field1 = Measured_amplitude * np.exp(1j * phase0)
#         else:
#             field1 = Measured_amplitude * np.exp(1j * update_phase[k - 1])
#             if k in noise_iterations:
#                 random_noise = 0.1 * (np.random.rand(*Measured_amplitude.shape) + 1j * np.random.rand(*Measured_amplitude.shape))
#                 field1 += random_noise
#         # b) back-propagation and apply energy constraint
#         field2 = angular_spectrum_method(field1, area, -distance, W, H, wavelength, min_frq, max_frq)
#         phase_field2 = np.angle(field2) # phase
#         amp_field2 = np.abs(field2) # amplitude
#         abso = -np.log(amp_field2)
#         # Apply constraints
#         abso[abso < 0] = 0
#         phase_field2[abso < 0] = 0
#         amp_field2 = np.exp(-abso)
#         field22 = amp_field2 * np.exp(1j * phase_field2)
#
#         # c) forward propagation and update amplitude
#         field3 = angular_spectrum_method(field22, area, distance, W, H, wavelength, min_frq, max_frq)
#         amp_field3 = np.abs(field3)
#         phase_field3 = np.angle(field3)
#         update_phase.append(phase_field3)
#         last_field = field3
#         # tell if next iteration is needed
#         if k > 0:
#             amp_diff = amp_field3 - Measured_amplitude
#             rms_error = np.sqrt(np.mean(amp_diff ** 2))
#             rms_errors.append(rms_error)
#             print(f"the {k} iteration, Error RMS {rms_error}")
#             if rms_error < convergence_threshold:  # 小于阈值，认为已收敛
#                 print(f"Converged at iteration {k}")
#                 # field_final = Norm_amplitude * np.exp(1j * phase_field3)
#                 return last_field
#     # Plot RMS error curve after the iteration ends
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(1, len(rms_errors) + 1), rms_errors, marker='o', linewidth=0.8)
#     plt.title("RMS Error Over Iterations")
#     plt.xlabel("Iteration")
#     plt.ylabel("RMS Error")
#     plt.grid()
#     plt.show()
#     return last_field
#
# # find the image
#
# field_ite = IPR(hologram_amplitude, z, wavelength, 500, 1e-20, area, W, H, min_frq, max_frq)
# IPR_object = angular_spectrum_method(field_ite, area, -z, W, H, wavelength, min_frq, max_frq)
# plot_field(IPR_object)

