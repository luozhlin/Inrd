addpath assess_fold
addpath NGmeet

filename = 'KSC_denoise_70.mat';
data_dir = './data';
data_path = fullfile(data_dir, filename);
load(data_path);

results_dir = './results';
save_path = fullfile(results_dir, filename);

[M,N,p] = size(gt);
Par   = ParSetH(sigma,p);
[output_image]= NGmeet_DeNoising(255*input, 255*gt, Par);  %NGmeet denoisng function    

[NGmeet_PSNR,NGmeet_SSIM,NGmeet_SAM,NGmeet_MQ] = evaluate(gt,output_image/255,M,N);
disp(['Method Name:NGmeet    ', ', MPSNR=' num2str(mean(NGmeet_PSNR),'%5.2f')  ...
           ',MSSIM = ' num2str(mean(NGmeet_SSIM),'%5.4f')  ',SAM=' num2str(NGmeet_SAM,'%5.2f')...
           ',MQ=' num2str(mean(NGmeet_MQ),'%5.4f')]);

result = output_image/255;

save(save_path, 'result');

disp(['数据已保存至: ', save_path]);
