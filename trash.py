import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2

# --- load sample ---
def load_and_normalize_image(filepath):
    image = Image.open(filepath).convert('L')  # 转灰度
    grayscale_data = np.array(image, dtype=np.float32)
    return (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())

object = load_and_normalize_image('pic/microscopic_sample_no_grid.png')

# --- Angular spectrum method ---
def Transfer_function(W, H, distance, wavelength, pixelSize, numPixels):
    FX = W / (pixelSize * numPixels)
    FY = H / (pixelSize * numPixels)
    k = 2 * np.pi / wavelength
    square_root = np.sqrt(1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2))
    valid_mask = (wavelength ** 2 * FX ** 2 + wavelength ** 2 * FY ** 2) <= 1
    square_root[~valid_mask] = 0
    temp = np.exp(1j * k * distance * square_root)
    return temp


def angular_spectrum_method(field, pixelSize, distance, W, H, numPixels):
    GT = fftshift(fft2(ifftshift(field)))
    transfer = Transfer_function(W, H, distance, 532e-9, pixelSize, numPixels)
    gt_prime = fftshift(ifft2(ifftshift(GT * transfer)))
    return gt_prime

def fourier_downsample(image, new_shape):

    N = image.shape[0]  # N x N
    M = new_shape[0]
    # 1. Calculate the Fourier frequency
    F = np.fft.fftshift(np.fft.fft2(image))
    # 2. Downsize the support area
    start = (N - M) // 2
    end = start + M
    F_crop = F[start:end, start:end]
    # 3. Inverse transform
    image_down = np.fft.ifft2(np.fft.ifftshift(F_crop))
    image_down = np.abs(image_down)
    image_down *= (N / M) ** 2
    return image_down

# --- Setting pixel sizes ---
sensor_pixel_sizes = [0.2e-6, 0.8e-6, 1.6e-6, 2.4e-6]  # 0.2µm, 0.8µm, 1.6µm
numPixels_original = 1024  # 原始分辨率
FOV = numPixels_original * sensor_pixel_sizes[0]  # 固定视场范围
z2 = 0.005  # 传播距离

# --- Deal with different sensors ---
for pixelSize in sensor_pixel_sizes:
    numPixels = int(FOV / pixelSize)  # 计算新分辨率
    x = (np.arange(numPixels) - numPixels / 2 - 1)
    y = (np.arange(numPixels) - numPixels / 2 - 1)
    W, H = np.meshgrid(x, y)

    object_down = fourier_downsample(object, (numPixels, numPixels))

    # --- Field of sample ---
    am = np.exp(-0.1 * object_down)
    ph0 = 3
    ph = ph0 * object_down
    field_after_object = am * np.exp(1j * ph)
    amp_field_after = np.abs(field_after_object)
    plt.figure(figsize=(6, 6))
    plt.imshow(amp_field_after, cmap='gray')
    plt.colorbar(label="Reconstructed Amplitude")
    plt.title(f"field after sample for {pixelSize}")
    plt.axis('off')
    plt.show()


    # --- Calculate the hologram ---
    hologram_field = angular_spectrum_method(field_after_object, pixelSize, z2, W, H, numPixels)
    print(hologram_field.size)
    hologram_amplitude = np.abs(hologram_field)

    plt.figure(figsize=(6, 6))
    plt.imshow(hologram_amplitude, cmap='gray')
    plt.colorbar(label="Reconstructed Amplitude")
    plt.title(f"hologram (Pixel Size: {pixelSize}µm)")
    plt.axis('off')
    plt.show()

    print(f"Saved hologram with pixel size {pixelSize}µm, resolution {numPixels}×{numPixels}")


    # --- Adding noise---
    # 1. Scaling (For IMX477, its full well capacity is 8000e-)
    scaling_factor = 8000
    ideal_intensity = (hologram_amplitude ** 2) * scaling_factor

    # 2. Shot Noise
    shot_noisy_intensity = np.random.poisson(ideal_intensity)
    shot_noisy_intensity = shot_noisy_intensity / scaling_factor

    # 3. Dark current noise
    thermal_noise_std = 1 / scaling_factor
    thermal_noise = np.random.normal(0, thermal_noise_std, ideal_intensity.shape)

    # 4. Readout Noise
    readout_noise_std = 3 / scaling_factor
    readout_noise = np.random.normal(0, readout_noise_std, ideal_intensity.shape)

    # 5. Quantization noise/ACD noise
    # For IMX477 its bit depth of output can be 8/10/12, we assume we use 12 bit depth
    quant_step_size = 2 ** 12 - 1
    hologram_quantized = np.round(ideal_intensity * quant_step_size) / quant_step_size
    quant_error = hologram_quantized - ideal_intensity

    # 6. Sum up the noise
    total_intensity = shot_noisy_intensity + thermal_noise + readout_noise + quant_error

    # Make sure amplitude >= 0
    total_intensity[total_intensity < 0] = 0

    # 6. Convert to intensity form
    hologram_amplitude_noisy = np.sqrt(total_intensity)


    # --- IPR ---
    def IPR(Measured_amplitude, distance, k_max, convergence_threshold, pixelSize, W, H, numPixels, amp_field_after):
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
            field2 = angular_spectrum_method(field1, pixelSize, -distance, W, H, numPixels)
            phase_field2 = np.angle(field2)  # phase
            amp_field2 = np.abs(field2)  # amplitude
            abso = -np.log(amp_field2)
            # Apply constraints
            abso[abso < 0] = 0
            phase_field2[abso < 0] = 0
            amp_field2 = np.exp(-abso)
            field22 = amp_field2 * np.exp(1j * phase_field2)


            # c) Forward propagation
            field3 = angular_spectrum_method(field22, pixelSize, distance, W, H, numPixels)
            amp_field3 = np.abs(field3)
            phase_field3 = np.angle(field3)
            update_phase.append(phase_field3)


            field4 = angular_spectrum_method(field3, pixelSize, -distance, W, H, numPixels)
            amp_field4 = np.abs(field4)
            last_field = field4
            # Error calculation
            if k > 0:
                rms_error = np.sqrt(np.mean((amp_field4 - amp_field_after) ** 2))
                rms_errors.append(rms_error)
                print(f"Iteration {k}: RMS Error = {rms_error}")

                ssim_value = ssim(amp_field_after, amp_field4, data_range=Measured_amplitude.max() - Measured_amplitude.min())
                ssim_errors.append(ssim_value)
                print(f"Iteration {k}: SSIM = {ssim_value}")

                # threshold
                if rms_error < convergence_threshold:
                    print(f"Converged at iteration {k}")
                    return last_field, rms_errors, ssim_errors
        # Draw RMS
        plt.subplot(2, 1, 1)
        plt.plot(rms_errors, 'r-', linewidth=2, label='RMS Error')
        plt.title('Convergence Analysis')
        plt.ylabel('RMS Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Draw SSIM
        plt.subplot(2, 1, 2)
        plt.plot(ssim_errors, 'b-', linewidth=2, label='SSIM')
        plt.xlabel('Iteration')
        plt.ylabel('SSIM')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig('convergence_metrics.png', dpi=300)
        plt.show()
        return last_field, rms_errors, ssim_errors

    # --- Run IPR ---
    field_ite, rms_errors, ssim_errors = IPR(hologram_amplitude_noisy, z2, 80, 1.5e-20, pixelSize, W, H, numPixels, amp_field_after)


    # --- Draw image ---
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(field_ite), cmap='gray')
    plt.colorbar(label="Reconstructed Amplitude")
    plt.title(f"Reconstructed Image (Pixel Size: {pixelSize*1e6}µm)")
    plt.axis('off')
    plt.show()

    print(f"Reconstruction for pixel size {pixelSize*1e6}µm completed.\n")
