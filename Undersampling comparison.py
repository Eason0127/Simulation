import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Simplified of test1. With fewer filter on it
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

def load_and_normalize_image(filepath):
    # Load the image
    image = Image.open(filepath).convert('L')  # Convert to grayscale
    # Convert image to a NumPy array
    grayscale_data = np.array(image, dtype=np.float32)
    # Normalize the grayscale data to [0, 1]
    normalized_data = (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())
    return normalized_data

def Transfer_function(W, H, distance, wavelength, area):
    FX = W / area
    FY = H / area
    square_root = np.sqrt(1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2))
    valid_mask = (wavelength ** 2 * FX ** 2 + wavelength ** 2 * FY ** 2) <= 1
    square_root[~valid_mask] = 0
    temp = np.exp(1j * 2 * np.pi * distance / wavelength * square_root)
    # plot_FX_FY(FX, FY)
    return temp

def angular_spectrum_method(field, area, distance, W, H):
    GT = fftshift(fft2(ifftshift(field)))
    gt_prime = fftshift(ifft2(ifftshift(GT * Transfer_function(W, H, distance, 532e-9, area))))
    return gt_prime


numPixels = 128
pixelSize = 1.6e-6 # unit: meter
z2 = 0.005
area = numPixels * pixelSize
# Define the sensor grid
x = np.arange(numPixels) - numPixels / 2 - 1
y = np.arange(numPixels) - numPixels / 2 - 1
W, H = np.meshgrid(x, y)


# Define the field after sample
hologram_amplitude = np.load('resized_data.npy').astype(np.float32)  # 读取时转换回来
plot_field(hologram_amplitude)


# IPR
def IPR(Measured_amplitude, distance, k_max, convergence_threshold, area, W, H):
    update_phase = []
    last_field = None
    rms_errors = []  # Store RMS errors for plotting
    ssim_errors = []
    for k in range(k_max):
        # a) sensor plane
        if k == 0:
            phase0 = np.zeros(Measured_amplitude.shape)
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

        # c) forward propagation and update amplitude
        field3 = angular_spectrum_method(field22, area, distance, W, H)
        amp_field3 = np.abs(field3)
        phase_field3 = np.angle(field3)
        update_phase.append(phase_field3)
        last_field = field3
        # tell if next iteration is needed
        if k > 0:
            amp_diff = amp_field3 - Measured_amplitude
            rms_error = np.sqrt(np.mean(amp_diff ** 2))
            rms_errors.append(rms_error)
            print(f"the {k} iteration, Error RMS {rms_error}")
            ssim_value = ssim(Measured_amplitude, amp_field3, data_range=Measured_amplitude.max() - Measured_amplitude.min())
            ssim_errors.append(ssim_value)
            print(f"the {k} iteration, Error SSIM {ssim_value}")

            # if rms_error < convergence_threshold:  # 小于阈值，认为已收敛
            #     print(f"Converged at iteration {k}")
            #     # field_final = Norm_amplitude * np.exp(1j * phase_field3)
            #     return last_field
    # 绘制RMS误差曲线
    plt.subplot(2, 1, 1)
    plt.plot(rms_errors, 'r-', linewidth=2, label='RMS Error')
    plt.title('Convergence Analysis')
    plt.ylabel('RMS Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 绘制SSIM曲线
    plt.subplot(2, 1, 2)
    plt.plot(ssim_errors, 'b-', linewidth=2, label='SSIM')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('convergence_metrics.png', dpi=300)
    plt.show()

    return last_field

# find the image

field_ite = IPR(hologram_amplitude, z2, 450, 1.5e-8, area, W, H)
IPR_object = angular_spectrum_method(field_ite, area, -z2, W, H)
plot_field(IPR_object)





