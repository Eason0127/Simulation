import numpy as np
import matplotlib.pyplot as plt


# Digital Holography Phase Retrieval

def iterative_phase_retrieval_holography(measured_intensity, support, wavelength, pixel_size, z_distance,
                                         iterations=100):
    """
    Iterative phase retrieval for digital holography using Gerchberg-Saxton algorithm.

    Parameters:
        measured_intensity (ndarray): Measured intensity (hologram).
        support (ndarray): Support constraint in the spatial domain (binary mask).
        wavelength (float): Wavelength of the light (in meters).
        pixel_size (float): Pixel size (in meters).
        z_distance (float): Propagation distance (in meters).
        iterations (int): Number of iterations to perform.

    Returns:
        ndarray: Reconstructed complex field in the spatial domain.
    """
    # Calculate Fourier coordinates
    size = measured_intensity.shape[0]
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(size, d=pixel_size)
    fy = np.fft.fftfreq(size, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z_distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))

    # Initialize random phase
    phase = np.random.rand(*measured_intensity.shape) * 2 * np.pi
    complex_field = np.sqrt(measured_intensity) * np.exp(1j * phase)

    for _ in range(iterations):
        # Forward propagation to hologram plane
        hologram_field = np.fft.fft2(complex_field) * H

        # Enforce intensity constraint in the hologram plane
        hologram_field = np.sqrt(measured_intensity) * np.exp(1j * np.angle(hologram_field))

        # Backward propagation to object plane
        complex_field = np.fft.ifft2(hologram_field * np.conj(H))

        # Apply support constraint
        complex_field[support == 0] = 0

    return complex_field


# Example usage
if __name__ == "__main__":
    # Parameters
    size = 256
    wavelength = 532e-9  # Green light (532 nm)
    pixel_size = 6.5e-6  # 6.5 micrometers per pixel
    z_distance = 0.01  # Propagation distance: 1 cm
    iterations = 200

    # Create a test object (spatial domain)
    true_object = np.zeros((size, size))
    true_object[size // 4:3 * size // 4, size // 4:3 * size // 4] = 1.0

    # Simulate hologram intensity
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(size, d=pixel_size)
    fy = np.fft.fftfreq(size, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z_distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))

    hologram = np.fft.fft2(true_object) * H
    measured_intensity = np.abs(hologram) ** 2

    # Define support constraint
    support = np.zeros((size, size))
    support[size // 4:3 * size // 4, size // 4:3 * size // 4] = 1

    # Perform iterative phase retrieval
    reconstructed_field = iterative_phase_retrieval_holography(measured_intensity, support, wavelength, pixel_size,
                                                               z_distance, iterations)

    # Visualize results
    plt.figure(figsize=(12, 6))

    # True object
    plt.subplot(1, 3, 1)
    plt.title("True Object")
    plt.imshow(np.abs(true_object), cmap="gray")
    plt.colorbar()

    # Measured intensity
    plt.subplot(1, 3, 2)
    plt.title("Measured Intensity")
    plt.imshow(np.log(1 + measured_intensity), cmap="gray")
    plt.colorbar()

    # Reconstructed object
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Object")
    plt.imshow(np.abs(reconstructed_field), cmap="gray")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
