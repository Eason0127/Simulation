import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom


# --- 设定不同像素尺寸的传感器 ---
sensor_pixel_sizes = [0.2e-6, 0.8e-6, 1.6e-6, 2.4e-6]  # 0.2µm, 0.8µm, 1.6µm
numPixels_original = 1024  # 原始分辨率
FOV = numPixels_original * sensor_pixel_sizes[0]  # 固定视场范围
z2 = 0.005  # 传播距离

# --- 加载 & 归一化样本 ---
def load_and_normalize_image(filepath):
    image = Image.open(filepath).convert('L')  # 转灰度
    grayscale_data = np.array(image, dtype=np.float32)
    return (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())

object = load_and_normalize_image('pic/microscopic_sample_no_grid.png')

# --- 角谱传播方法 ---
def Transfer_function(W, H, distance, wavelength, pixelSize, numPixels):
    FX = W / (pixelSize * numPixels)
    FY = H / (pixelSize * numPixels)
    k = 2 * np.pi / wavelength
    square_root = np.sqrt(1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2))
    valid_mask = (wavelength ** 2 * FX ** 2 + wavelength ** 2 * FY ** 2) <= 1
    square_root[~valid_mask] = 0
    temp = np.exp(1j * k * distance * square_root)
    return temp


def resize_transfer_function(transfer, new_shape):
    zoom_factors = (new_shape[0] / transfer.shape[0], new_shape[1] / transfer.shape[1])

    # 先使用最近邻插值，避免新值
    resized = zoom(transfer, zoom_factors, order=0)

    # 让所有插值的值变为 0（保持原始点）
    mask = zoom(np.ones_like(transfer), zoom_factors, order=0)  # 原始点为1，其它点为0
    resized = resized * mask  # 只保留原始点，其余变0
    return resized

def angular_spectrum_method(field, pixelSize, distance, W, H, numPixels):
    GT = fftshift(fft2(ifftshift(field)))
    transfer = Transfer_function(W, H, distance, 532e-9, pixelSize, numPixels)
    transfer = resize_transfer_function(transfer, field.shape)
    gt_prime = fftshift(ifft2(ifftshift(GT * transfer)))
    return gt_prime



# --- 生成样本场 ---
am = np.exp(-1.6 * object)
ph0 = 3
ph = ph0 * object
field_after_object = am * np.exp(1j * ph)
amp_field_after = np.abs(field_after_object)
plt.figure(figsize=(6, 6))
plt.imshow(amp_field_after, cmap='gray')
plt.colorbar(label="Reconstructed Amplitude")
plt.title(f"field after sample")
plt.axis('off')
plt.show()

# --- 处理不同像素大小的传感器 ---
for pixelSize in sensor_pixel_sizes:
    numPixels = int(FOV / pixelSize)  # 计算新分辨率
    x = (np.arange(numPixels) - numPixels / 2 - 1)
    y = (np.arange(numPixels) - numPixels / 2 - 1)
    W, H = np.meshgrid(x, y)


    # --- 计算全息图 ---
    hologram_field = angular_spectrum_method(field_after_object, pixelSize, z2, W, H, numPixels)
    print(hologram_field.size)
    hologram_amplitude = np.abs(hologram_field)

    plt.figure(figsize=(6, 6))
    plt.imshow(hologram_amplitude, cmap='gray')
    plt.colorbar(label="Reconstructed Amplitude")
    plt.title(f"hologram (Pixel Size: {pixelSize*1e6}µm)")
    plt.axis('off')
    plt.show()

    # --- 保存全息图数据 ---
    # np.save(f"hologram_{int(pixelSize*1e9)}nm.npy", hologram_amplitude.astype(np.float32))
    print(f"Saved hologram with pixel size {pixelSize*1e6}µm, resolution {numPixels}×{numPixels}")

    # --- IPR 过程 ---
    def IPR(Measured_amplitude, distance, k_max, convergence_threshold, pixelSize, W, H, numPixels, amp_field_after):
        update_phase = []
        last_field = None
        rms_errors = []
        ssim_errors = []

        for k in range(k_max):
            # a) 传感器平面
            if k == 0:
                phase0 = np.zeros(Measured_amplitude.shape)
                field1 = Measured_amplitude * np.exp(1j * phase0)
            else:
                field1 = Measured_amplitude * np.exp(1j * update_phase[k - 1])

            # b) 反向传播并施加能量约束
            field2 = angular_spectrum_method(field1, pixelSize, -distance, W, H, numPixels)
            phase_field2 = np.angle(field2)  # phase
            amp_field2 = np.abs(field2)  # amplitude
            abso = -np.log(amp_field2)
            # Apply constraints
            abso[abso < 0] = 0
            phase_field2[abso < 0] = 0
            amp_field2 = np.exp(-abso)
            field22 = amp_field2 * np.exp(1j * phase_field2)

            # c) 正向传播并更新振幅
            field3 = angular_spectrum_method(field22, pixelSize, distance, W, H, numPixels)
            amp_field3 = np.abs(field3)
            phase_field3 = np.angle(field3)
            update_phase.append(phase_field3)

            field4 = angular_spectrum_method(field3, pixelSize, -distance, W, H, numPixels)
            amp_field4 = np.abs(field4)
            last_field = field4
            # 计算误差
            if k > 0:
                rms_error = np.sqrt(np.mean((amp_field4 - amp_field_after) ** 2))
                rms_errors.append(rms_error)
                print(f"Iteration {k}: RMS Error = {rms_error}")

                ssim_value = ssim(amp_field_after, amp_field4,
                                  data_range=Measured_amplitude.max() - Measured_amplitude.min())
                ssim_errors.append(ssim_value)
                print(f"Iteration {k}: SSIM = {ssim_value}")

                # 终止条件
                if rms_error < convergence_threshold:
                    print(f"Converged at iteration {k}")
                    return last_field, rms_errors, ssim_errors
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
        return last_field, rms_errors, ssim_errors


    # --- 运行 IPR ---
    field_ite, rms_errors, ssim_errors = IPR(hologram_amplitude, z2, 80, 1.5e-20, pixelSize, W, H, numPixels,
                                             amp_field_after)

    # --- 绘制图像 ---
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(field_ite), cmap='gray')
    plt.colorbar(label="Reconstructed Amplitude")
    plt.title(f"Reconstructed Image (Pixel Size: {pixelSize * 1e6}µm)")
    plt.axis('off')
    plt.show()

    print(f"Reconstruction for pixel size {pixelSize * 1e6}µm completed.\n")
