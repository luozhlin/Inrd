clear,clc

addpath functions
addpath functions/TV_operator
addpath assess_fold
% % Demo on i.i.d. Gaussian Noise
filename = 'WHU_Hi_LongKou_denoise_30.mat';

data_dir = './data';
data_path = fullfile(data_dir, filename);
load(data_path);

results_dir = './results';
save_path = fullfile(results_dir, filename);


Ori_H = gt;
[M, N, B] = size(Ori_H);
Noi_H = input;

 %% WETV
disp('############## WETV #################')
j=10    
param.Rank   = [7,7,5];
param.initial_rank = 2;
param.maxIter = 50;
param.lambda    = 4e-3*sqrt(M*N);  %0.1;   %mu2      = Alpha*mu1
tic
[output_image,U_x,V_x,E] = WETV(Noi_H,Ori_H, param);
Re_hsi_wETV   = reshape(output_image,M,N,B);
Time  = toc;

% evaluation
[NGmeet_PSNR,NGmeet_SSIM,NGmeet_SAM,NGmeet_MQ] = evaluate(gt,Re_hsi_wETV,M,N);
disp(['Method Name:ETPTV    ', ', MPSNR=' num2str(mean(NGmeet_PSNR),'%5.2f')  ...
           ',MSSIM = ' num2str(mean(NGmeet_SSIM),'%5.4f')  ',SAM=' num2str(NGmeet_SAM,'%5.2f')...
           ',MQ=' num2str(mean(NGmeet_MQ),'%5.4f')]);


% 储存数据
result = Re_hsi_wETV;
save(save_path, 'result');
disp(['数据已保存至: ', save_path]);
