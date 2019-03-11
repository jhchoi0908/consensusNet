clear all; clc; close all;
format compact;
global sigmas;

run /home/choi240/matconvnet-1.0-beta25/matlab/vl_setupnn
bm3d_path	= '../../BM3D';
dncnn_path	= '../../DnCNN';
ffdnet_path	= '../../FFDNet';
rednet_path	= '../../REDNet';

rng(1337, 'twister');

addpath(fullfile(bm3d_path));
addpath(fullfile(dncnn_path, 'utilities'));
addpath(fullfile(ffdnet_path, 'utilities'));
addpath(fullfile(rednet_path, 'caffe/matlab'));
addpath(fullfile(rednet_path, 'utils'));

folder_test     = '/depot/chan129/data/CSNet/BSD200';
filePaths   = [];
ext         = {'*.jpg', '*.png', '*.bmp'};
for i = 1 : length(ext)
    filePaths   = cat(1, filePaths, dir(fullfile(folder_test, ext{i})));
end
PSNR	= zeros(4, 5);

for sigma = 10:20:70
    
    i	= int16((sigma+10)/20);
    images_gt       = fullfile('/depot/chan129/data/CSNet/results/different_type', num2str(sigma,'%02d'), 'groundtruth');
    images_in       = fullfile('/depot/chan129/data/CSNet/results/different_type', num2str(sigma,'%02d'), 'inputs');
    images_dncnn    = fullfile('/depot/chan129/data/CSNet/results/different_type', num2str(sigma,'%02d'), 'dncnn');
    images_bm3d     = fullfile('/depot/chan129/data/CSNet/results/different_type', num2str(sigma,'%02d'), 'bm3d');
    images_ffdnet   = fullfile('/depot/chan129/data/CSNet/results/different_type', num2str(sigma,'%02d'), 'ffdnet');
    
    if ~exist(images_gt, 'dir')
        mkdir(images_gt);
    end
    if ~exist(images_in, 'dir')
        mkdir(images_in);
    end
    if ~exist(images_dncnn, 'dir')
        mkdir(images_dncnn);
    end
    if ~exist(images_bm3d, 'dir')
        mkdir(images_bm3d);
    end
    if ~exist(images_ffdnet, 'dir')
        mkdir(images_ffdnet);
    end
    
    folder_dncnn    = fullfile(dncnn_path, 'model/specifics', ['sigma=',num2str(sigma,'%02d'), '.mat']);
    folder_ffdnet   = fullfile(ffdnet_path, 'models', 'FFDNet_gray.mat');
    
    for j = 1:length(filePaths)
        
        gt      = im2double(imread(fullfile(folder_test, filePaths(j).name)));
        if size(gt, 3) > 1
            image = rgb2gray(image);
        end
        [h, w]  = size(gt);
%        imwrite(gt,  fullfile(images_gt, filePaths(i).name));
        
        in      = gt + sigma/255.0*randn(size(gt));
%        newfile = replace(filePaths(i).name, {'.jpg', '.png', '.bmp'}, '.txt');
%        dlmwrite(fullfile(images_in, newfile), in, 'delimiter', ' ')
        PSNR(i, 1) = PSNR(i, 1) + 10*log10(1/mean(mean((in-gt).^2)))
        
        %BM3D
        [~, out]    = BM3D(1, in, sigma);
%        imwrite(out,  fullfile(images_bm3d, filePaths(i).name));
        PSNR(i, 2) = PSNR(i, 2) + 10*log10(1/mean(mean((out-gt).^2)))
        
        % DnCNN
        load(fullfile(folder_dncnn));
        in_gpu  = gpuArray(single(in));
        net     = vl_simplenn_tidy(net);
        net     = vl_simplenn_move(net, 'gpu');
        res     = vl_simplenn(net, in_gpu, [], [], 'conserveMemory', true, 'mode', 'test');
        out     = in_gpu - res(end).x;
        
        out     = double(gather(out));
%        imwrite(out,  fullfile(images_dncnn, filePaths(i).name));
        PSNR(i, 3) = PSNR(i, 3) + 10*log10(1/mean(mean((out-gt).^2)))
        
        %FFDNet
        in_gpu  = single(in);
        if mod(h,2)==1
            in_gpu  = cat(1, in_gpu, in_gpu(end,:));
        end
        if mod(w,2)==1
            in_gpu  = cat(2, in_gpu, in_gpu(:,end));
        end
        
        in_gpu  = gpuArray(in_gpu);
        load(fullfile(folder_ffdnet));
        net     = vl_simplenn_tidy(net);
        net     = vl_simplenn_move(net, 'gpu');
        sigmas  = sigma/255;
        res     = vl_simplenn(net, in_gpu, [], [], 'conserveMemory', true, 'mode', 'test');
        out     = res(end).x;
        
        if mod(h,2)==1
            out     = out(1:end-1, :);
        end
        if mod(w,2)==1
            out     = out(:, 1:end-1);
        end
        
        out     = double(gather(out));
%        imwrite(out,  fullfile(images_ffdnet, filePaths(i).name));
        PSNR(i, 4) = PSNR(i, 4) + 10*log10(1/mean(mean((out-gt).^2)))
        
        % REDNet
        in_gpu  = single(in);
        out     = netforward(in_gpu, sigma, 'denoising');
        PSNR(i, 5) = PSNR(i, 5) + 10*log10(1/mean(mean((out-gt).^2)))
    end
end
PSNR = PSNR./length(filePaths)
disp(PSNR)

