clear; clc;
global sigmas;

run /home/matconvnet-1.0-beta25/matlab/vl_setupnn
dncnn_path	= '';
bm3d_path	= '';
ffdnet_path	= '';

rng(1337, 'twister');

addpath(fullfile(dncnn_path, 'utilities'));
addpath(fullfile(bm3d_path, 'BM3D'));

num_denoisers   = 4;
modelSigma      = 20;

folder_test = '../data';
folder_gt   = fullfile(folder_test, 'groundtruth');
folder_in   = fullfile(folder_test, 'inputs');
folder_out  = fullfile(folder_test, 'combined');
folder_dncnn    = fullfile(dncnn_path, 'model/specifics', ['sigma=',num2str(modelSigma,'%02d'), '.mat']);
folder_ffdnet   = fullfile(ffdnet_path, 'models', 'model_gray.mat');

filePaths   = [];
ext         = {'*.jpg', '*.png', '*.bmp'};
for i = 1 : length(ext)
    filePaths   = cat(1, filePaths, dir(fullfile(folder_gt, ext{i})));
end

MSE     = zeros(length(filePaths), num_denoisers);
MSE_    = zeros(length(filePaths), num_denoisers);
weights = zeros(length(filePaths), num_denoisers);
outs    = cell(num_denoisers, 1);
PSNRs   = zeros(length(filePaths), num_denoisers+1);

fileID  = fopen(fullfile(folder_test, 'sigma.txt'), 'r');
keyval  = textscan(fileID, '%s %f');
dict_sigma  = containers.Map(keyval{1}', keyval{2}');

folder_tmp  = fullfile(folder_in, 'rednet');
fileID  = fopen(fullfile(folder_tmp, 'mse.txt'), 'r');
keyval  = textscan(fileID, '%s %f');
dict_rednet = containers.Map(keyval{1}', keyval{2}');

for i = 1:length(filePaths)
    
    sigma   = dict_sigma(filePaths(i).name);
    epsilon = sigma/100/255;
    
    gt      = im2double(imread(fullfile(folder_gt, filePaths(i).name)));
    in      = im2double(imread(fullfile(folder_in, filePaths(i).name)));
    [h, w]  = size(gt);
    
    b       = randn(size(in));
    in_     = in + epsilon*b;
    
    k       = 0;
    
    % DnCNN
    k       = k + 1;
    load(fullfile(folder_dncnn));
    
    in_gpu  = gpuArray(single(in));
    in_gpu_ = gpuArray(single(in_));
    
    net     = vl_simplenn_move(net, 'gpu');
    res     = vl_simplenn(net, in_gpu, [], [], 'conserveMemory', true, 'mode', 'test');
    out     = in_gpu - res(end).x;
    
    res     = vl_simplenn(net, in_gpu_, [], [], 'conserveMemory', true, 'mode', 'test');
    out_    = in_gpu_ - res(end).x;
    
    outs{k} = double(gather(out));
    out_    = double(gather(out_));
    
    divF    = (b(:)'*(out_(:)-outs{k}(:)))/(numel(in)*epsilon);
    MSE(i, k)  = mean( (in(:) - outs{k}(:)).^2 ) - (sigma/255)^2 + (2*(sigma/255)^2)*divF;
    imwrite(outs{k},  fullfile(folder_in, 'dncnn', filePaths(i).name));
    
    %FFDNet
    k       = k + 1;
    
    in_gpu  = single(in);
    in_gpu_ = single(in_);
    
    if mod(h,2)==1
        in_gpu  = cat(1, in_gpu, in_gpu(end,:));
        in_gpu_ = cat(1, in_gpu_, in_gpu_(end,:));
    end
    if mod(w,2)==1
        in_gpu  = cat(2, in_gpu, in_gpu(:,end));
        in_gpu_ = cat(2, in_gpu_, in_gpu_(:,end));
    end
    
    in_gpu  = gpuArray(in_gpu);
    in_gpu_ = gpuArray(in_gpu_);
    
    load(fullfile(folder_ffdnet));
    net     = vl_simplenn_tidy(net);
    net     = vl_simplenn_move(net, 'gpu');
    sigmas  = sigma/255;
    res     = vl_simplenn(net, in_gpu, [], [], 'conserveMemory', true, 'mode', 'test');
    out     = in_gpu - res(end).x;
    
    res     = vl_simplenn(net, in_gpu_, [], [], 'conserveMemory', true, 'mode', 'test');
    out_    = in_gpu_ - res(end).x;
    
    if mod(h,2)==1
        out     = out(1:end-1, :);
        out_    = out_(1:end-1, :);
    end
    if mod(w,2)==1
        out     = out(:, 1:end-1);
        out_    = out_(:, 1:end-1);
    end
    
    outs{k} = double(gather(out));
    out_    = double(gather(out_));
    
    divF    = (b(:)'*(out_(:)-outs{k}(:)))/(numel(in)*epsilon);
    MSE(i, k)  = mean( (in(:) - outs{k}(:)).^2 ) - (sigma/255)^2 + (2*(sigma/255)^2)*divF;
    imwrite(outs{k},  fullfile(folder_in, 'ffdnet', filePaths(i).name));
    
    %BM3D
    k       = k + 1;
    [~, outs{k}]    = BM3D(1, in, sigma);
    [~, out_]       = BM3D(1, in_, sigma);
    
    divF    = (b(:)'*(out_(:)-outs{k}(:)))/(numel(in)*epsilon);
    MSE(i, k)  = mean( (in(:) - outs{k}(:)).^2 ) - (sigma/255)^2 + (2*(sigma/255)^2)*divF;
    imwrite(outs{k},  fullfile(folder_in, 'bm3d', filePaths(i).name));
    
    %REDNet
    k       = k + 1;
    MSE(i, k)   = dict_rednet(filePaths(i).name);
    outs{k} = im2double(imread(fullfile(folder_tmp, filePaths(i).name)));
    
    %SURE
    minMSE  = min(MSE(i, :));
    for j=1:num_denoisers
        MSE_(i, j) = MSE(i, j) - minMSE;
    end
    sigma_p	= std(MSE_(i, :));
    
    for j=1:num_denoisers
        weights(i, j)  = exp(-MSE_(i, j) / (0.3*sigma_p));
    end
    
    weights_sum     = sum(weights(i, :));
    for j=1:num_denoisers
        weights(i, j)  = weights(i, j)/weights_sum;
    end
    
    output  = zeros(h, w);
    for j=1:num_denoisers
        output  = output + outs{j}*weights(i, j);
    end
    
    %Evaluate
    for j=1:num_denoisers
        PSNRs(i, j) = 10*log10(1/mean((gt(:)-outs{j}(:)).^2));
    end
    
    mu_out  = mean(output(:));
    PSNRs(i, num_denoisers+1) = 10*log10(1/mean((gt(:)-output(:)).^2));
    imwrite(output,  fullfile(folder_test, 'combined', filePaths(i).name));
    if mod(i, 20)==0
        disp(i);
    end
end
disp(mean(PSNRs));

