import os
import numpy as np
import hdf5storage
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import scipy.io as sio


def load_mat_file(file_path, key='croppedData'):
    """加载mat文件并返回指定key的数据"""
    try:
        data = hdf5storage.loadmat(file_path)
        return data[key]
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {str(e)}")
        return None

def calculate_channel_metrics(original, reconstructed, data_range=1.0):
    """计算每个通道的PSNR和SSIM，返回平均值和各通道值"""
    # 确保输入形状一致
    if original.shape != reconstructed.shape:
        raise ValueError(f"形状不匹配: 原始数据 {original.shape}, 重建数据 {reconstructed.shape}")

    # 确保数据类型为float
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    # 获取通道数（假设最后一维是通道）
    num_channels = original.shape[-1] if original.ndim == 3 else 1
    psnr_values = []
    ssim_values = []

    for channel in range(num_channels):
        # 提取当前通道
        if original.ndim == 3:
            orig_chan = original[..., channel]
            rec_chan = reconstructed[..., channel]
        else:
            orig_chan = original
            rec_chan = reconstructed

        # 计算PSNR
        psnr_val = psnr(orig_chan, rec_chan, data_range=data_range)
        psnr_values.append(psnr_val)

        # 计算SSIM
        ssim_val = ssim(orig_chan, rec_chan, data_range=data_range)
        ssim_values.append(ssim_val)

    # 计算平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim, psnr_values, ssim_values

def main():
    # 配置路径（用户可能需要根据实际情况调整）
    result_dir = './images_result/wdc_denoise_50'
    data_dir = './data'  # 根据项目结构选择合适的数据目录
    data_key = 'gt'       # mat文件中的数据关键字
    data_range = 1.0               # 数据范围（0-1或0-255等）
    data_path = os.path.join(data_dir, "wdc_denoise_50.mat")
    
    original_data = sio.loadmat(data_path)
    gt = original_data[data_key]
    print(gt.shape)
    
    result_path = os.path.join(result_dir, "tdsat.mat")
    res_data = sio.loadmat(result_path)
    print(res_data.keys())
    res = res_data["R_hsi"]
    print(res.shape)
    # 获取所有结果文
        # 计算指标
    print(f"GT数据范围: [{np.min(gt)}, {np.max(gt)}]")
    print(f"GT数据类型: {gt.dtype}")
    print(f"结果数据范围: [{np.min(res)}, {np.max(res)}]")

    avg_psnr, avg_ssim, psnr_channels, ssim_channels = calculate_channel_metrics(
                gt, res, data_range)

    print(f"平均PSNR: {avg_psnr:.4f}")
    print(f"平均SSIM: {avg_ssim:.4f}")


if __name__ == '__main__':
    main()