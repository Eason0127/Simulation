import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fft, fftfreq
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.signal import find_peaks
import os
import re
import math
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

## --- Read image and normalization ---
def load_and_normalize_image(filepath):
    image = Image.open(filepath).convert('L')
    grayscale_data = np.array(image, dtype=np.float32)
    return (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())

## --- Plot image ---

def plot_image2(amplitude,title):
    plt.figure(figsize=(6, 6))
    plt.imshow(amplitude, cmap='gray')
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_image(amplitude,title, save_dir, pixel, picture):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(amplitude, cmap='gray')
    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_title(title)
    ax.axis('off')
    # —— 保存 ——
    if save_dir is not None and pixel is not None and picture is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{pixel*1e6:.2f}um-{picture*1e6:.2f}um.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 图像已保存到：{save_path}")
    else:
        # 如果不满足保存条件，仅关闭 figure
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

## --- Set pitch size of the image and sensor ---
sensor_pixel_sizes = np.arange(1.6, 3, 0.02) * 1e-6  # The range of pixel size from 0.2-3 micrometer and step size 0.05
spacing_um = np.arange(5, 20, 0.5) * 1e-6
resolutions = [] # Store the reconstruction result
for i in range (len(sensor_pixel_sizes)):
    FOV_initial = 409.6e-6
    numPixels_sensor = int(FOV_initial // sensor_pixel_sizes[i])  # The dimension of the image
    FOV = numPixels_sensor * sensor_pixel_sizes[i] # The real FOV
    z2 = 0.0005  # Sample to sensor distance
    wavelength = 532e-9  # Wavelength
    ## Select the image's pixel size
    candidates = [0.2e-6, 0.1e-6, 0.02e-6]
    for p in candidates:
        q = sensor_pixel_sizes[i] / p
        if abs(q - round(q)) < 1e-6: 
            image_pixel_size = p
            factor = int(round(q))
            break
    else:
        raise ValueError(f"No matching image_pixel_size for sensor pixel {sensor_pixel_sizes[i]}")
    print(sensor_pixel_sizes[i], image_pixel_size)

    factor = int(sensor_pixel_sizes[i] / image_pixel_size + 0.5)
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
        region_size = img_size // 2
        start = (img_size - region_size) // 2
        end = start + region_size
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

        # --- Define the spatial grid of sample plane ---
        x = np.arange(numPixel_sample) - numPixel_sample / 2 - 1
        y = np.arange(numPixel_sample) - numPixel_sample / 2 - 1
        W, H = np.meshgrid(x, y)

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
        hologram_field = angular_spectrum_method(object_field, image_pixel_size, z2, W, H, numPixel_sample, wavelength, f_cut)
        in_hologram = np.abs(hologram_field) ** 2
        am_hologram = np.sqrt(in_hologram)

        # --- Calculate the dimension of sampled hologram ---
        sampled_hologram = am_hologram[::factor, ::factor]
        am_object_field_down = am_object_field[::factor, ::factor]

        # --- Create the sensor grid ---
        numPixels_sensor2 = sampled_hologram.shape[0]
        x_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
        y_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
        W_sen, H_sen = np.meshgrid(x_sen, y_sen)

        ## --- Code check ---
        print(f"Factor = {factor}")
        print(f"sample shape = {object_shape}")
        print(f"hologram shape = {numPixels_sensor2}")

        # --- Reconstruction based on IPR algo ---
        rec_field, rms_errors, ssim_errors = IPR(sampled_hologram, z2, 50, 1.5e-20, sensor_pixel_sizes[i], W_sen, H_sen, numPixels_sensor, am_object_field_down, wavelength, f_cut)
        am_rec_field = np.abs(rec_field)
        plot_image(am_rec_field, "rec field", r'/Users/wangmusi/Desktop/Research/Reconstruction relationship/pixel2', sensor_pixel_sizes[i], n)

        # --- Contrast ---
        sample_size_sensor = numPixels_sensor # 传感器平面的图像像素数
        region_size_sensor = sample_size_sensor // 2
        start_sensor = (sample_size_sensor - region_size_sensor) // 2
        end_sensor = start_sensor + region_size_sensor
        region = am_rec_field[start_sensor:end_sensor, start_sensor:end_sensor]
        # 1) Read the value on the lines
        y_indices = [5,10,15,20]
        x_indices = [-10,-5,5,10]
        contrasts = []
        ## 上半区域的对比度
        for y in y_indices:
            line_vals = region[y, :]
            PSF = line_vals ** 2
            ## Plot PSF
            x = (np.arange(PSF.size) - PSF.size // 2) * sensor_pixel_sizes[i] * 1e6
            plt.figure(figsize=(6, 4))  # <-- 新建一个 figure
            plt.plot(x, PSF, linewidth=2)
            plt.xlabel('Position (μm)')
            plt.ylabel('PSF Intensity')
            plt.title(f'PSF Profile, sensor pitch = {sensor_pixel_sizes[i] * 1e6:.2f}μm, spacing = {n * 1e6:.2f}μm')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            ## 2) Find the maximum of the bright stripe and the minimum of the dark stripe.
            try:
                period_sensor = grating_period // factor
                stripe_width_sensor = stripe_width // factor
                size = PSF.size

                # 跳过不合理情况
                if stripe_width_sensor < 1 or period_sensor <= stripe_width_sensor or size < period_sensor:
                    raise ValueError("窗口太小，跳过")

                n_periods = size // period_sensor
                PSF_cut = PSF[:n_periods * period_sensor]
                print(f"图像周期：{grating_period}，传感器周期：{period_sensor}，图像条纹{stripe_width}，传感器条纹{stripe_width_sensor}")
                I_max = []
                I_min = []
                for k in range(n_periods):
                    trough_block = PSF_cut[k * period_sensor: k * period_sensor + stripe_width_sensor]
                    peak_block = PSF_cut[k * period_sensor + stripe_width_sensor: (k + 1) * period_sensor]
                    I_min.append(trough_block.min())
                    I_max.append(peak_block.max())
                for k in range(len(I_max) - 1):
                    I_peak = min(I_max[k], I_max[k + 1])
                    I_trough = I_min[k]
                    contrasts.append((I_peak - I_trough) / (I_peak + I_trough))
                    contrast_test = []
                    contrast_test.append((I_peak - I_trough) / (I_peak + I_trough)) # 计算每个PSF上的对比度均值，方便检查
                contrast1 = np.mean(contrast_test)
                print(f"🐶{y}处的PSF上的对比度的值为{contrast1}.")
            except Exception as e:
                # 这里捕获上面抛出的跳过或其它错误，直接 continue
                print(f"Contrast 计算时出错 ({e})，跳过 n={n * 1e6:.2f}μm")
                continue
        ## 下半区域的对比度
        for x in x_indices:
            m = region_size_sensor // 2
            line_vals = region[0:m , x]
            PSF = line_vals ** 2
            x = (np.arange(PSF.size) - PSF.size // 2) * sensor_pixel_sizes[i] * 1e6
            plt.figure(figsize=(6, 4))  # <-- 新建一个 figure
            plt.plot(x, PSF, linewidth=2)
            plt.xlabel('Position (μm)')
            plt.ylabel('PSF Intensity')
            plt.title(f'PSF Profile, sensor pitch = {sensor_pixel_sizes[i] * 1e6:.2f}μm, spacing = {n * 1e6:.2f}μm')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            try:
                period_sensor = grating_period // factor
                stripe_width_sensor = stripe_width // factor
                size = PSF.size
                # 跳过不合理情况
                if stripe_width_sensor < 1 or period_sensor <= stripe_width_sensor or size < period_sensor:
                    raise ValueError("窗口太小，跳过")

                n_periods = size // period_sensor
                PSF_cut = PSF[:n_periods * period_sensor]
                print(f"图像周期：{grating_period}，传感器周期：{period_sensor}，图像条纹{stripe_width}，传感器条纹{stripe_width_sensor}")
                I_max = []
                I_min = []
                for k in range(n_periods):
                    trough_block = PSF_cut[k * period_sensor: k * period_sensor + stripe_width_sensor]
                    peak_block = PSF_cut[k * period_sensor + stripe_width_sensor: (k + 1) * period_sensor]
                    I_min.append(trough_block.min())
                    I_max.append(peak_block.max())
                for k in range(len(I_max) - 1):
                    I_peak = min(I_max[k], I_max[k + 1])
                    I_trough = I_min[k]
                    contrasts.append((I_peak - I_trough) / (I_peak + I_trough))
                    contrast_test = []
                    contrast_test.append((I_peak - I_trough) / (I_peak + I_trough)) # 计算每个PSF上的对比度均值，方便检查
                contrast1 = np.mean(contrast_test)
                print(f"🐶{x}处的PSF上的对比度的值为{contrast1}.")
            except Exception as e:
                # 这里捕获上面抛出的跳过或其它错误，直接 continue
                print(f"Contrast 计算时出错 ({e})，跳过 n={n * 1e6:.2f}μm")
                continue
        ## 计算总的对比度            
        contrast_mean = np.mean(contrasts)
        ## 判断条件
        if contrast_mean > 0.25:
            print(f"🌟[i={i}] Sensor pixel = {sensor_pixel_sizes[i] * 1e6:.2f}μm: "f"80% contrasts >0.25，resolvable at {n * 1e6:.2f}μm")
            resolutions.append(n)
            break
        else:
            print(f"🚩[i={i}] Contrast value = {contrast_mean:.2f}, it's not resolvable at {n * 1e6:.2f}μm")             

# 转成 μm 单位方便阅读
sensor_pitches_um = sensor_pixel_sizes * 1e6
resolutions_um = np.array(resolutions) * 1e6

plt.figure(figsize=(6,4))
plt.plot(sensor_pitches_um, resolutions_um, linestyle='-')
plt.xlabel('Sensor pitch (μm)')
plt.ylabel('Resolved stripe period (μm)')
plt.title('Resolution vs. Sensor Pitch Size')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存
out_path = 'resolution_vs_sensor_pitch.png'
plt.savefig(out_path, dpi=300)
out_path = r'/Users/wangmusi/Desktop/Research/Reconstruction relationship/resolution_vs_sensor_pitch.png'
plt.savefig(out_path, dpi=300)
print(f"✅ Plot saved to {out_path}")

plt.show()
