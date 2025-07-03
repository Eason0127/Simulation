import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fft, fftfreq
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.signal import find_peaks
import os
import re

# --- Read image and normalization ---
def load_and_normalize_image(filepath):
    image = Image.open(filepath).convert('L')
    grayscale_data = np.array(image, dtype=np.float32)
    return (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())

# --- Plot image ---
def plot_image(amplitude,title, save_dir, pixel, picture):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(amplitude, cmap='gray')
    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_title(title)
    ax.axis('off')
    plt.show()
    # —— 保存 ——
    if save_dir is not None and pixel is not None and picture is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{pixel}-{picture}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 图像已保存到：{save_path}")
    else:
        # 如果不满足保存条件，仅关闭 figure
        plt.close(fig)

# --- Filter image ---
def bandlimit_filter(image, pixelSize):
    N = image.shape[0]
    F = fftshift(fft2(image))
    k = np.arange(N) - N / 2
    f = k / (N * pixelSize)
    FX, FY = np.meshgrid(f, f)
    f_magnitude = np.sqrt(FX ** 2 + FY ** 2)
    #  Nyquist frequency
    f_max = 1 / (2 * pixelSize)
    mask = f_magnitude <= f_max
    F_filtered = F * mask
    image_filtered = ifft2(ifftshift(F_filtered))
    return image_filtered

# --- Transfer function + filter evanescent wave ---
def Transfer_function(W, H, distance, wavelength, pixelSize, numPixels, f_cut):
    FX = W / (pixelSize * numPixels)
    FY = H / (pixelSize * numPixels)
    k = 2 * np.pi / wavelength
    a = 1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2)
    square_root = np.sqrt(np.clip(a, 0, None))
    Hf = np.exp(1j * k * distance * square_root)
    # Evanescent wave filtering
    valid_mask = a >= 0 # Evanescent wave filtering
    NA_mask = FX ** 2 + FY ** 2 <= f_cut ** 2 # NA limitation filtering
    total_mask = valid_mask & NA_mask
    Hf[~total_mask] = 0
    return Hf

# --- Angular spectrum method ---
def angular_spectrum_method(field, pixelSize, distance, W, H, numPixels, wavelength, f_cut):
    GT = fftshift(fft2(ifftshift(field)))
    transfer = Transfer_function(W, H, distance, wavelength, pixelSize, numPixels, f_cut)
    gt_prime = fftshift(ifft2(ifftshift(GT * transfer)))
    return gt_prime

# --- IPR ---
def IPR(Measured_amplitude, distance, k_max, convergence_threshold, pixelSize, W, H, numPixels, amp_field_after, wavelength, f_cut):
    update_phase = []
    last_field = None
    rms_errors = []
    ssim_errors = []

    for k in range(k_max):
        # a) Sensor plane
        if k == 0:
            phase0 = np.zeros(Measured_amplitude.shape)
            field1 = Measured_amplitude * np.exp(1j * phase0)
        else:
            field1 = Measured_amplitude * np.exp(1j * update_phase[k - 1])
        # b) Backpropagation and apply constraint
        field2 = angular_spectrum_method(field1, pixelSize, -distance, W, H, numPixels, wavelength, f_cut)
        phase_field2 = np.angle(field2)  # phase
        amp_field2 = np.abs(field2)  # amplitude
        abso = -np.log(amp_field2 + 1e-8) #1e-8 to prevent 0 value
        # Apply constraints
        abso[abso < 0] = 0
        phase_field2[abso < 0] = 0
        amp_field2 = np.exp(-abso)
        field22 = amp_field2 * np.exp(1j * phase_field2)
        # c) Forward propagation
        field3 = angular_spectrum_method(field22, pixelSize, distance, W, H, numPixels, wavelength, f_cut)
        amp_field3 = np.abs(field3)
        phase_field3 = np.angle(field3)
        update_phase.append(phase_field3)

        # d) Backpropagate to get the image
        field4 = angular_spectrum_method(field3, pixelSize, -distance, W, H, numPixels, wavelength, f_cut)
        amp_field4 = np.abs(field4)
        last_field = field4
        # Error calculation
        if k > 0:
            rms_error = np.sqrt(np.mean((amp_field_after - amp_field4) ** 2))
            rms_errors.append(rms_error)
            print(f"Iteration {k}: RMS Error = {rms_error}")

            ssim_value = ssim(amp_field_after, amp_field4, data_range=amp_field_after.max() - amp_field_after.min())
            ssim_errors.append(ssim_value)
            print(f"Iteration {k}: SSIM = {ssim_value}")

            # threshold
            if rms_error < convergence_threshold:
                print(f"Converged at iteration {k}")
                return last_field, rms_errors, ssim_errors
    # Draw RMS
    # plt.subplot(2, 1, 1)
    # plt.plot(rms_errors, 'r-', linewidth=2, label='RMS Error')
    # plt.title('Convergence Analysis')
    # plt.ylabel('RMS Error')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # # Draw SSIM
    # plt.subplot(2, 1, 2)
    # plt.plot(ssim_errors, 'b-', linewidth=2, label='SSIM')
    # plt.xlabel('Iteration')
    # plt.ylabel('SSIM')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('convergence_metrics.png', dpi=300)
    # plt.show()
    return last_field, rms_errors, ssim_errors

#----------------------------------------Divided Line-------------------------------------------

# --- Read image ---
file_path = '/Users/wangmusi/Desktop/1.png'
file_name = os.path.basename(file_path)
m = re.search(r'(\d+(?:\.\d+)?)_test', file_name)
value = float(m.group(1))
print(value)
object = load_and_normalize_image(file_path)
period = int(value * 1e-6 / 0.2e-6)

# --- Set pixel size of the image and sensor ---
sensor_pixel_sizes = [0.2e-6, 2e-6]  # 0.2µm for image, 1.6µm for sensor
numPixels_image = 242  # The dimension of the image
FOV = numPixels_image * sensor_pixel_sizes[0]  # Calculate image's FOV
z2 = 0.001  # Sample to sensor distance
wavelength = 532e-9 # Wavelength

# --- Define the spatial grid ---
x = np.arange(numPixels_image) - numPixels_image / 2 - 1
y = np.arange(numPixels_image) - numPixels_image / 2 - 1
W, H = np.meshgrid(x, y)

# --- Filter the image before forward propagation ---
# object_filtered = bandlimit_filter(object, sensor_pixel_sizes[1])
# am_object_filtered = np.abs(object_filtered)
# am_object = np.abs(object)

# ---Define the sample field ---
am = np.exp(-2 * object)
ph0 = 3
ph = ph0 * object
object_field = am * np.exp(1j * ph)
am_object_field = np.abs(object_field)

# --- The Filtering issue due to NA limitation ---
# This cut-off frequency is input into the transfer function
NA = (FOV / 2) / np.sqrt((FOV / 2) ** 2 + z2 ** 2) # Numerical Aperture
f_cut = NA / wavelength # The lateral frequency on sensor plane


# --- Acquire the hologram ---
hologram_field = angular_spectrum_method(object_field, sensor_pixel_sizes[0], z2, W, H, numPixels_image, wavelength, f_cut)
in_hologram = np.abs(hologram_field) ** 2
am_hologram = np.sqrt(in_hologram)

# --- Calculate the dimension of sampled hologram ---
undersample_factor = int(sensor_pixel_sizes[1] / sensor_pixel_sizes[0])
sampled_hologram_size = am_hologram[::undersample_factor, ::undersample_factor]
am_object_field_down = am_object_field[::undersample_factor, ::undersample_factor]

# --- Create the sensor grid ---
numPixels_sensor = sampled_hologram_size.shape[0]
x_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
y_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
W_sen, H_sen = np.meshgrid(x_sen, y_sen)

# --- Downsample the hologram based on the sensor pixel size ---
undersample_factor = int(sensor_pixel_sizes[1] / sensor_pixel_sizes[0])
am_undersampled_hologram = am_hologram[::undersample_factor, ::undersample_factor]
am_object_field_down = am_object_field[::undersample_factor, ::undersample_factor]
Sampled_hologram = am_undersampled_hologram ** 2

# --- Pixel aperture effect ---
# FX = W / (numPixels_image * sensor_pixel_sizes[0])
# FY = H / (numPixels_image * sensor_pixel_sizes[0])
# Pixel_TF = np.sinc(FX * sensor_pixel_sizes[1]) * np.sinc(FY * sensor_pixel_sizes[1])
# hologram_fft = fftshift(fft2(ifftshift(hologram_field)))
# hologram_fft_window = hologram_fft * Pixel_TF
# hologram_field_filtered = fftshift(ifft2(ifftshift(hologram_fft_window)))
# center = np.arange(undersample_factor//2, numPixels_image, undersample_factor)
# Sampled_hologram_field = hologram_field_filtered[center[:,None], center]
# Sampled_hologram = np.abs(Sampled_hologram_field) ** 2

# --- Adding noise ---
# scaling_factor = 8000 # Assume full well capacity is 8000e-
# ideal_intensity = (am_undersampled_hologram ** 2) * scaling_factor # Transform intensity to the scale of photons or electrons
# noise_electrons = 0 # Choose the number of noise electrons
# noise_standard = noise_electrons / scaling_factor # Transform noise form scale of electrons to scale of intensity
# white_Gaussian_noise = np.random.normal(0, noise_standard, ideal_intensity.shape) # Simulate the noise
# am_noise = np.abs(white_Gaussian_noise)
# # plot_image(am_noise)
# am_hologram_with_noise = am_undersampled_hologram + white_Gaussian_noise

# --- Calculate SNR ---
# noise_power = np.mean(white_Gaussian_noise ** 2)
# signal_power = np.mean(am_undersampled_hologram ** 2)
# SNR = 10 * np.log10(signal_power / noise_power)


# --- Reconstruction based on IPR algo ---
rec_field, rms_errors, ssim_errors = IPR(Sampled_hologram, z2, 50, 1.5e-20, sensor_pixel_sizes[1], W_sen, H_sen, numPixels_sensor, am_object_field_down, wavelength, f_cut)
am_rec_field = np.abs(rec_field)
plot_image(am_rec_field, "rec field",'/Users/wangmusi/Desktop', z2, value)

# --- Contrast ---
# y_index = 65
# value = am_rec_field[y_index, :]
# PSF = np.abs(value) ** 2
# plt.figure(figsize=(8,4))
# plt.plot(x_sen, PSF, linewidth=2)
# plt.xlabel('x (m)')
# plt.ylabel('Intensity')
# plt.title('PSF Profile')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
# peaks, _ = find_peaks(PSF, height = None, distance = period * 0.7)
# troughs, _ = find_peaks(-PSF, height = None, distance = period * 0.7)
# I_max = PSF[peaks].mean()
# I_min = PSF[troughs].mean()
# print(f"I_max = {I_max}, I_min = {I_min}")
# C = (I_max - I_min) / (I_max + I_min)
# print(f"Contrast = {C}")
# if C >= 0.1:
#     print("It's resolved!")
# else:
#     print("It's not resolved!")

# --- PSF ---
# y_index = 86
# PSF_pre = am_rec_field[y_index, :]
# PSF = np.abs(PSF_pre) ** 2
# x_axis = x_sen
# # plot
# plt.figure(figsize=(8,4))
# plt.plot(x_axis, PSF, linewidth=2)
# plt.xlabel('x')
# plt.ylabel('Intensity')
# plt.title(f'PSF')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # --- MTF ---
# Np = PSF.size
# print(Np)
# OTF1d = fftshift(fft(ifftshift(PSF)))         # 1D OTF
# MTF1d = np.abs(OTF1d)
# MTF1d /= MTF1d.max()
# F_x = x_axis / (Np * sensor_pixel_sizes[1])
# plt.figure(figsize=(6,4))
# plt.plot(F_x, MTF1d, linewidth=2)
# plt.xlabel('Spatial frequency (1/m)')
# plt.ylabel('MTF')
# plt.title('1D MTF from PSF profile')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


