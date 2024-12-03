import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image


def load_hologram(filename, shape=None):
    """
    Load hologram data from a file and reshape it if a shape is provided.
    Supports binary and image files.
    """
    if filename.endswith('.bin'):
        # Load binary file
        data = np.fromfile(filename, dtype=np.float32)
        print(f"Loaded binary data size: {data.size}")
    elif filename.endswith(('.jpg', '.png', '.jpeg')):
        # Load image file
        image = Image.open(filename).convert('L')
        data = np.array(image, dtype=np.float32)
        print(f"Loaded image data size: {data.size}")
    else:
        raise ValueError("Unsupported file format. Use .bin or image formats like .jpg, .png")

    if shape is not None:
        # Ensure the provided shape is compatible with the data size
        if np.prod(shape) != data.size:
            raise ValueError(f"Cannot reshape data of size {data.size} into shape {shape}")
        hologram = data.reshape(shape)
    else:
        # Try to infer square shape if shape is not provided
        size = int(np.sqrt(data.size))
        if size * size == data.size:
            hologram = data.reshape((size, size))
        else:
            raise ValueError("Data size is not suitable for reshaping into a square.")

    return hologram

def initialize_field(hologram):
    """
    Initialize the complex-valued field at the detector plane.
    """
    measured = np.sqrt(hologram)  # Calculate amplitude
    measured = (measured - measured.min()) / (measured.max() - measured.min())  # Normalize to range [0, 1]
    phase = np.zeros_like(hologram)  # Initial phase is zero
    field_detector = measured * np.exp(1j * phase)
    return measured, field_detector

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
    temp[np.isnan(temp)] = 0  # replace nan's with zeros
    return temp
def angular_spectrum_method(field, area, distance, W, H):
    GT = ifftshift(fft2(fftshift(field)))
    gt_prime = fftshift(ifft2(ifftshift(GT * Transfer_function(W, H, distance, 532e-9, area))))
    return gt_prime



numPixels = 500
shape = (numPixels, numPixels)
pixelSize = 1e-7 # unit: meter
area = numPixels * pixelSize
x = np.arange(numPixels) - numPixels / 2 - 1
y = np.arange(numPixels) - numPixels / 2 - 1
W, H = np.meshgrid(x, y)


# Define the field after sample
hologram = load_hologram('b_hologram.jpg', shape)
plot_field(hologram)
Measured_amp, field_detector = initialize_field(hologram)




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

field_ite = IPR(Measured_amp, 0.05, 10000, 1e-4, area, W, H)
plot_field(field_ite)

