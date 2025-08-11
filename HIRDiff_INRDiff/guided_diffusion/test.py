import torch

# 假设输入图像的形状为 (256, 256, 5)
image = torch.rand(256, 256, 5)  # 示例数据
num_channels = image.size(2)  # 获取通道数

# 动态生成分割规则
splits = []
for i in range(num_channels):
    # 生成 3 个连续的通道索引（循环）
    split_indices = [(i + j) % num_channels for j in range(3)]
    splits.append(split_indices)

# 对每个分割组合进行操作（例如计算均值）
processed_splits = []
for split in splits:
    # 提取分割的图像
    split_image = image[:, :, split]  # 形状为 (256, 256, 3)
    # 对分割的图像进行操作（这里以计算均值为例）
    processed_split = torch.mean(split_image, dim=2, keepdim=True)  # 在第 3 维度上取均值，形状为 (256, 256, 1)
    processed_splits.append(processed_split)

# 初始化结果矩阵
result = torch.zeros_like(image)  # 形状为 (256, 256, 5)

# 合并结果并在对应位置上取平均
for channel in range(num_channels):
    # 找到所有包含当前 channel 的分割组合
    contributing_splits = []
    for i, split in enumerate(splits):
        if channel in split:
            # 找到当前 channel 在 split 中的位置
            pos_in_split = split.index(channel)
            # 提取对应的值
            contributing_value = processed_splits[i][:, :, 0]  # 形状为 (256, 256)
            contributing_splits.append(contributing_value)
    
    # 将所有贡献的值堆叠并取平均
    if contributing_splits:
        stacked_values = torch.stack(contributing_splits, dim=0)  # 形状为 (N, 256, 256)
        result[:, :, channel] = torch.mean(stacked_values, dim=0)  # 在第 0 维度上取平均

print(result.shape)  # 输出结果的形状 (256, 256, 5)