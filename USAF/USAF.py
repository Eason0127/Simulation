import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import downscale_local_mean


def load_and_normalize_image(filepath):
    image = Image.open(filepath).convert('L')
    grayscale_data = np.array(image, dtype=np.float32)
    return (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())

def plot_image(amplitude,):
    plt.figure(figsize=(6, 6))
    plt.imshow(amplitude, cmap='gray')
    plt.colorbar(label="Amplitude")
    plt.title(f"Amplitude")
    plt.axis('off')
    plt.show()

def bandlimit_filter(image, pixelSize):
    N = image.shape[0]
    F = fftshift(fft2(image))
    k = np.arange(N) - N / 2
    f = k / (N * pixelSize)
    FX, FY = np.meshgrid(f, f)
    f_magnitude = np.sqrt(FX ** 2 + FY ** 2)
    #  Nyquist frequency
    f_max = 312500 # 1 / (2 * pixelSize)
    mask = f_magnitude <= f_max
    F_filtered = F * mask
    image_filtered = ifft2(ifftshift(F_filtered))
    return image_filtered

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
        abso = -np.log(amp_field2 + 1e-8) #1e-8 to prevent 0 value
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

        # d) Backpropagate to get the image
        field4 = angular_spectrum_method(field3, pixelSize, -distance, W, H, numPixels)
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

#----------------------------------------Divided Line-------------------------------------------

# --- Read image ---
object = load_and_normalize_image('')
plt.figure(figsize=(6, 6))
plt.imshow(object, cmap='gray')
plt.title("Converted USAF Target")
plt.axis('off')
plt.show()

# --- Set pixel size of the image and sensor ---
sensor_pixel_sizes = [0.2e-6, 1.2e-6]  # 1µm for image, 1.6µm for sensor
numPixels_image = 10560  # The dimension of the image
FOV = numPixels_image * sensor_pixel_sizes[0]  # Calculate image's FOV
z2 = 0.001  # Sample to sensor distance

# --- Define the spatial grid ---
x = np.arange(numPixels_image) - numPixels_image / 2 - 1
y = np.arange(numPixels_image) - numPixels_image / 2 - 1
W, H = np.meshgrid(x, y)

# --- Filter the image before forward propagation ---
object_filtered = bandlimit_filter(object, sensor_pixel_sizes[1])
am_object_filtered = np.abs(object_filtered)
am_object = np.abs(object)

# --- Plot the two images ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(am_object, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(am_object_filtered, cmap='gray')
plt.title("Bandlimited Image")
plt.axis('off')
plt.show()

# ---Define the sample field ---
am = np.exp(-1.6 * object_filtered)
ph0 = 3
ph = ph0 * object_filtered
object_field = am * np.exp(1j * ph)
am_object_field = np.abs(object_field)
plot_image(am_object_field)

# --- Acquire the hologram ---
hologram_field = angular_spectrum_method(object_field, sensor_pixel_sizes[0], z2, W, H, numPixels_image)
am_hologram = np.abs(hologram_field)
# plot_image(am_hologram)

# --- Downsample the hologram based on the sensor pixel size ---
undersample_factor = int(sensor_pixel_sizes[1] / sensor_pixel_sizes[0])
am_undersampled_hologram = downscale_local_mean(am_hologram, (undersample_factor, undersample_factor))
# plot_image(am_undersampled_hologram)
am_object_field_down = downscale_local_mean(am_object_field, (undersample_factor, undersample_factor))
plot_image(am_object_field_down)


# --- Create the sensor grid ---
numPixels_sensor = am_undersampled_hologram.shape[0]
x_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
y_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
W_sen, H_sen = np.meshgrid(x_sen, y_sen)

# --- Adding noise ---
# At first, I will just consider the white Guassian noise. If nothing wrong then I will go deeper.
scaling_factor = 8000 # Assume full well capacity is 8000e-
ideal_intensity = (am_undersampled_hologram ** 2) * scaling_factor # Transform intensity to the scale of photons or electrons
noise_electrons = 0 # Choose the number of noise electrons
noise_standard = noise_electrons / scaling_factor # Transform noise form scale of electrons to scale of intensity
white_Gaussian_noise = np.random.normal(0, noise_standard, ideal_intensity.shape) # Simulate the noise
am_noise = np.abs(white_Gaussian_noise)
plot_image(am_noise)
am_hologram_with_noise = am_undersampled_hologram + white_Gaussian_noise

# --- Calculate SNR ---
noise_power = np.mean(white_Gaussian_noise ** 2)
signal_power = np.mean(am_undersampled_hologram ** 2)
SNR = 10 * np.log10(signal_power / noise_power)


# --- Reconstruction based on IPR algo ---
rec_field, rms_errors, ssim_errors = IPR(am_hologram_with_noise, z2, 400, 1.5e-20, sensor_pixel_sizes[1], W_sen, H_sen, numPixels_sensor, am_object_field_down)
am_rec_field = np.abs(rec_field)
plot_image(am_rec_field)
print("SNR is %f dB" % SNR)


