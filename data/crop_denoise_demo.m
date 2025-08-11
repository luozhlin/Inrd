% 伪彩图
function false_color = create_false_color_image(data)
% 创建假彩色图像 (使用近红外、红和绿波段)
[~, ~, bands] = size(data);

% 选择近红外、红和绿波段
nir_band = round(bands * 0.75);  % 近红外波段
r_band = round(bands * 0.55);    % 红波段
g_band = round(bands * 0.45);    % 绿波段

% 构建假彩色图像 (NIR-R-G)
false_color = cat(3, data(:,:,nir_band), data(:,:,r_band), data(:,:,g_band));

% 对比度增强 - 修改这里以正确处理RGB图像
false_color = imadjust(false_color, stretchlim(false_color));
end

% 加载数据
input = double(wdc);
% 未裁剪数据
inputn = (input - min(input(:)))/(max(input(:))- min(input(:)));

% 裁剪数据
h = 200;
w = 1;
input_crop = input(h:h+255, w:w+255, :);
input_crop = (input_crop - min(input_crop(:)))/(max(input_crop(:))- min(input_crop(:)));


f1 = create_false_color_image(inputn);
f2 = create_false_color_image(input_crop);
subplot(1,2,1)
imshow(f1)
subplot(1,2,2)
imshow(f2)


% 加噪
% sigma=30;
% noise = (sigma/255) * randn(size(gt));
% input = gt + noise;
% save("WHU_Hi_LongKou_denoise_110.mat", "gt", "input","sigma")
