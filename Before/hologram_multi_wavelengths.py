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


def angular_spectrum_method(field, sampling_distance, distance, FX, FY, wavelength):
    GT = ifftshift(fft2(fftshift(field))) * sampling_distance ** 2
    # * dx ** 2 :Make the discrete Fourier transform result numerically close to the continuous Fourier transform result and Maintain power consistency
    gt_prime = fftshift(ifft2(ifftshift(GT * Transfer_function(FX, FY, distance, wavelength)))) / sampling_distance ** 2
    return gt_prime


def compute_hologram_with_bandwidth(field, sampling_distance, distance, FX, FY, center_wavelength, bandwidth, num_samples):
    wavelengths = np.linspace(center_wavelength - bandwidth / 2, center_wavelength + bandwidth / 2, num_samples)
    hologram = np.zeros_like(field, dtype=np.complex128)

    for wavelength in wavelengths:
        hologram += angular_spectrum_method(field, sampling_distance, distance, FX, FY, wavelength)

    # Take the intensity (amplitude squared)
    hologram_intensity = np.abs(hologram) ** 2
    return hologram_intensity


# Parameters
numPixels = 512
pixelSize = 1e-7  # microns
Sample_Radius = 50  # In pixels
Sample_Phase = 0.75
Center = numPixels / 2  # assumes numPixels is even

W, H = np.meshgrid(np.arange(0, numPixels), np.arange(0, numPixels))  # coordinates of the array indexes
Mask = np.sqrt((W - Center) ** 2 + (H - Center) ** 2) <= Sample_Radius  # boundaries of the object

incident_field = np.ones((numPixels, numPixels), dtype=complex)
incident_field[Mask] = np.exp(1j * Sample_Phase)
plot_field(incident_field, title="Incident Field")

x = np.linspace(-pixelSize * numPixels / 2, pixelSize * numPixels / 2, num=numPixels, endpoint=True)
dx = x[1] - x[0]  # Sampling period
fS = 1 / dx  # Spatial sampling frequency
df = fS / numPixels  # Spacing between discrete frequency coordinates
fx = np.arange(-fS / 2, fS / 2, step=df)  # Spatial frequency, inverse microns
FX, FY = np.meshgrid(fx, fx)

# Broadband source parameters
center_wavelength = 532e-9  # 532 nm
bandwidth = 10e-9  # 10 nm
num_samples = 50  # Number of discrete wavelengths to sample

# Compute hologram
hologram = compute_hologram_with_bandwidth(incident_field, dx, 1e-5, FX, FY, center_wavelength, bandwidth, num_samples)

# Plot hologram
fig, ax = plt.subplots(nrows=1, ncols=1)
img = ax.imshow(hologram, cmap='hot')
img.set_extent([x[0], x[-1], x[0], x[-1]])
cb = plt.colorbar(img)
cb.set_label('Amplitude$^2$ (Broadband)')
ax.set_xticks([x[0], x[-1]])
ax.set_yticks([x[0], x[-1]])
ax.set_xlabel(r'$x, \, \mu m$')
ax.set_ylabel(r'$y, \, \mu m$')
plt.show()


