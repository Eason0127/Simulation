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
    GT = ifftshift(fft2(fftshift(field)))
    # * dx ** 2 :Make the discrete Fourier transform result numerically close to the continuous Fourier transform result and Maintain power consistency
    gt_prime = fftshift(ifft2(ifftshift(GT * Transfer_function(FX, FY, distance, 532e-9))))
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
incident_field = np.ones((numPixels, numPixels), dtype=complex)
incident_field[Mask] = np.exp(1j * Sample_Phase)
plot_field(incident_field)

x = np.linspace(-pixelSize * numPixels / 2, pixelSize * numPixels / 2, num = numPixels, endpoint = True)
physicalRadius = Sample_Radius * pixelSize
dx = x[1] - x[0]    # Sampling period
fS = 1 / dx         # Spatial sampling frequency
df = fS / numPixels # Spacing between discrete frequency coordinates
fx = np.arange(-fS / 2, fS / 2, step = df) # Spatial frequency, inverse microns
FX, FY = np.meshgrid(fx, fx)

hologram_field = angular_spectrum_method(incident_field, dx, 1e-5, FX, FY)
plot_field(hologram_field)

fig, ax = plt.subplots(nrows=1, ncols=1)
img = ax.imshow(np.abs(hologram_field)**2)
img.set_extent([x[0], x[-1], x[0], x[-1]])
cb = plt.colorbar(img)
cb.set_label('amplitude$^2$')
ax.set_xticks([x[0], x[-1]])
ax.set_yticks([x[0], x[-1]])
ax.set_xlabel(r'$x, \, \mu m$')
ax.set_ylabel(r'$y, \, \mu m$')
plt.show()



