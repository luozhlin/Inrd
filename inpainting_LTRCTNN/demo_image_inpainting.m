% applying tnsor completion for image inpainting
%
%
% version 1.0 - 01/02/2019
% version 1.1 - 29/04/2021
%
% Written by Canyi Lu (canyilu@gmail.com)
% 
% References:
% Canyi Lu, Xi Peng, Yunchao Wei, Low-Rank Tensor Completion With a New Tensor 
% Nuclear Norm Induced by Invertible Linear Transforms. IEEE International 
% Conference on Computer Vision and Pattern Recognition (CVPR), 2019
%

clear;
addpath('data');
addpath("assess_fold");  

filename = 'KSC_inpainting_5.mat';
data_dir = './data';
data_path = fullfile(data_dir, filename);
load(data_path);

X = input; 
maxP = 1;
dimX = size(X);
[MM, N, B]     = size(X);
% sampling rate


omega = find(mask==1);

M = mask;

M2 = Frontal2Lateral(M); % each lateral slice is a channel of the image
% omega2 = zeros(dimX);
% Iones = ones(dimX);
omega2 = ones(dimX);
Iones = zeros(dimX);

omega2(omega) = Iones(omega);
omega2 = Frontal2Lateral(omega2);
% omega2 = find(omega2==0);
omega2 = find(omega2==0);
n3 = size(M2,3);

% transform.L = @fft; transform.l = n3; transform.inverseL = @ifft;
transform.L = @dct; transform.l = 1; transform.inverseL = @idct;
% L = dftmtx(n3); trransform.l = n3; transform.L = L;
% L = dct(eye(n3)); transform.l = 1; transform.L = L;
% L = RandOrthMat(n3); transform.l = 1; transform.L = L;

opts.DEBUG = 1;
Xhat2 = lrtc_tnn(M2,omega2,transform,opts);
Xhat2 = max(Xhat2,0);
Xhat2 = min(Xhat2,maxP);
Xhat2 = Lateral2Frontal(Xhat2); % each lateral slice is a channel of the image


[NGmeet_PSNR,NGmeet_SSIM,NGmeet_SAM,NGmeet_MQ] = evaluate(gt,Xhat2,MM,N);
disp(['Method Name:LRTC  ', ', MPSNR=' num2str(mean(NGmeet_PSNR),'%5.2f')  ...
           ',MSSIM = ' num2str(mean(NGmeet_SSIM),'%5.4f')  ',SAM=' num2str(NGmeet_SAM,'%5.2f')...
           ',MQ=' num2str(mean(NGmeet_MQ),'%5.4f')]);

result = Xhat2;
results_dir = './results';
save_path = fullfile(results_dir, filename);

save(save_path, 'result');
disp(['数据已保存至: ', save_path]);
