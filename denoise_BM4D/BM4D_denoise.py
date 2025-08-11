import bm4d
import scipy.io as sio
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 加载数据
data_dir = "./data"
file_name = "WHU_Hi_LongKou_denoise_70.mat"
file_path = os.path.join(data_dir, file_name)


df = sio.loadmat(file_path)

# use
gt = df["gt"]
input = df["input"]
sigma = df["sigma"]
y_hat = bm4d.bm4d(input, sigma/255)


psnr_value = psnr(gt, y_hat, data_range=1)
ssim_value = ssim(gt, y_hat, data_range=1)

print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.4f}")

# 储存数据
filename_without_ext = os.path.splitext(file_name)[0]
result_name = f"{filename_without_ext}_result.mat"
print(result_name)

results_dir = './results'
save_path = os.path.join(results_dir, result_name)
sio.savemat(save_path, {'result': y_hat})
print(f"结果已保存至: {save_path}")