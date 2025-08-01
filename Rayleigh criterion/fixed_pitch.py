import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fft, fftfreq
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.signal import find_peaks
import csv
import os
import re
import math
from PIL import Image



## --- Read image and normalization ---
def load_and_normalize_image(filepath):
    image = Image.open(filepath).convert('L')
    grayscale_data = np.array(image, dtype=np.float32)
    return (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())

## --- Plot image ---

def plot_image2(amplitude, title):
    # 转成 float，防止整数截断
    amp = amplitude.astype(np.float32)
    # 归一化到 [0,1]
    amp = (amp - amp.min()) / (amp.max() - amp.min())

    plt.figure(figsize=(6, 6))
    # 强制显示范围为 [0,1]
    im = plt.imshow(amp, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(im, label="Normalized amplitude")
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_image3(amplitude,title):
    plt.figure(figsize=(6, 6))
    plt.imshow(amplitude, cmap='gray')
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_image(amplitude, title, save_dir, pixel, picture):
    # 先归一化到 [0,1]
    amp = amplitude.astype(np.float32)
    amp = (amp - amp.min()) / (amp.max() - amp.min())

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(amp, cmap='gray', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Normalized amplitude")
    ax.set_title(title)
    ax.axis('off')

    # —— 保存 ——
    if save_dir is not None and pixel is not None and picture is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{pixel*1e3:.2f}mm-{picture*1e6:.2f}um.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 图像已保存到：{save_path}")
    else:
        plt.close(fig)

## --- Filter image ---
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

## --- Transfer function + filter evanescent wave ---
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

## --- Angular spectrum method ---
def angular_spectrum_method(field, pixelSize, distance, W, H, numPixels, wavelength, f_cut):
    GT = fftshift(fft2(ifftshift(field)))
    transfer = Transfer_function(W, H, distance, wavelength, pixelSize, numPixels, f_cut)
    gt_prime = fftshift(ifft2(ifftshift(GT * transfer)))
    return gt_prime

## --- IPR ---
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
        # amp_field3 = np.abs(field3)
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
            # print(f"Iteration {k}: RMS Error = {rms_error}")

            ssim_value = ssim(amp_field_after, amp_field4, data_range=amp_field_after.max() - amp_field_after.min())
            ssim_errors.append(ssim_value)
            # print(f"Iteration {k}: SSIM = {ssim_value}")

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
## --- Set pitch size of the image and sensor ---
z2 = np.arange(0.5, 100, 1) * 1e-3  # 样本—传感器间距
spacing_um = np.arange(5, 50, 1) * 1e-6 # 生成的光栅间距设置
resolutions = [] # Store the reconstruction result
sensor_pitch_size = 1e-6 # 传感器大小
num_sensor = 4000 # 像素数量
FOV = num_sensor * sensor_pitch_size # 视野大小
wavelength = 532e-9  # Wavelength
for i in range (len(z2)):
    ## Select the image's pixel size
    image_pixel_size = 0.5e-6
    factor = int(sensor_pitch_size / image_pixel_size + 0.5)
    numPixel_sample = int(FOV / image_pixel_size)
    # --- Generate the sample at the image plane---
    img_size = numPixel_sample
    # --- Generate the sample ---
    for n in spacing_um:
        grating_period = int(n / image_pixel_size)
        stripe_width = grating_period // 2
        # Create a blank (black) image
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        # Define the central square region
        region_size = img_size // 35
        start = region_size // 20
        end = start + region_size
        # 不要黑色为0
        background_level = 0
        img = np.full((img_size, img_size), background_level, dtype=np.uint8)
        # Draw vertical stripes in the top half
        for x in range(start, end):
            if ((x - start) // stripe_width) % 2 == 0:
                img[start:start + region_size // 2, x] = 255
        # Draw horizontal stripes in the bottom half
        for y in range(start + region_size // 2, end):
            if ((y - (start + region_size // 2)) // stripe_width) % 2 == 0:
                img[y, start:end] = 255
        object = img.astype(float) / 255.0
        object_shape = object.shape[0]
        plot_image3(object,"object")
        # --- Define the spatial grid of sample plane ---
        g = np.arange(numPixel_sample) - numPixel_sample / 2 - 1
        h = np.arange(numPixel_sample) - numPixel_sample / 2 - 1
        W, H = np.meshgrid(g, h)
        print("Finished1")
        # ---Define the sample field ---
        am = np.exp(-0.5 * object)
        ph0 = 3
        ph = ph0 * object
        object_field = am * np.exp(1j * ph)
        am_object_field = np.abs(object_field)
        print("Finished2")
        # --- The Filtering issue due to NA limitation ---
        # This cut-off frequency is input into the transfer function
        NA = (FOV / 2) / np.sqrt((FOV / 2) ** 2 + z2[i] ** 2) # Numerical Aperture
        f_cut = NA / wavelength # The lateral frequency on sensor plane
        # --- Acquire the hologram ---
        hologram_field = angular_spectrum_method(object_field, image_pixel_size, z2[i], W, H, numPixel_sample, wavelength, f_cut)
        in_hologram = np.abs(hologram_field) ** 2
        am_hologram = np.sqrt(in_hologram)
        am_hologram /= am_hologram.max()
        print("Finished3")
        # --- Calculate the dimension of sampled hologram ---
        sampled_hologram = am_hologram[::factor, ::factor]
        am_object_field_down = am_object_field[::factor, ::factor]

        # --- Create the sensor grid ---
        numPixels_sensor2 = sampled_hologram.shape[0]
        x_sen = np.arange(num_sensor) - num_sensor / 2 - 1
        y_sen = np.arange(num_sensor) - num_sensor / 2 - 1
        W_sen, H_sen = np.meshgrid(x_sen, y_sen)

        ## --- Code check ---
        # print(f"Factor = {factor}")
        # print(f"sample shape = {object_shape}")
        # print(f"hologram shape = {numPixels_sensor2}")

        # --- Reconstruction based on IPR algo ---
        rec_field, rms_errors, ssim_errors = IPR(sampled_hologram, z2[i], 5, 1.5e-20, sensor_pitch_size, W_sen, H_sen, num_sensor, am_object_field_down, wavelength, f_cut)
        am_rec_field = np.abs(rec_field)
        am_rec_field /= am_rec_field.max() # 归一化
        # Save the image
        sample_size_sensor = num_sensor # 传感器平面的图像像素数
        region_size_sensor = sample_size_sensor // 4
        start_sensor = (sample_size_sensor - region_size_sensor) // 2
        end_sensor = start_sensor + region_size_sensor
        region = am_rec_field[start_sensor:end_sensor, start_sensor:end_sensor]
        print("Finished4")
        plot_image(region,"rec field", r"/Users/wangmusi/Desktop/Research/new_rec_test/setup_test",z2[i], n)


# 转成 μm 单位方便阅读
# sensor_pitches_um = sensor_pixel_sizes * 1e6
# resolutions_um = np.array(resolutions) * 1e6
#
# plt.figure(figsize=(6,4))
# plt.plot(sensor_pitches_um, resolutions_um, linestyle='-')
# plt.xlabel('Sensor pitch (μm)')
# plt.ylabel('Resolved stripe period (μm)')
# plt.title('Resolution vs. Sensor Pitch Size')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
#
# # 保存
# out_path = 'resolution_vs_sensor_pitch2.png'
# plt.savefig(out_path, dpi=300)
# out_path = r'/Users/wangmusi/Desktop/Research/new_rec_test/1/setup_test.png'
# plt.savefig(out_path, dpi=300)
# print(f"✅ Plot saved to {out_path}")
#
# plt.show()
