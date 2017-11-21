# Consensus Neural Network (ConsensusNet)

## Integrating Disparate Sources of Experts for Robust Image Denoising

### Abstract

We study an image denoising problem: Given a set of image denoisers, each having a different denoising capability, can we design a framework that allows us to integrate the individual denoisers to produce an overall better result? If we can do so, then potentially we can integrate multiple weak denoisers to denoise complex scenes. The goal of this paper is to present a meta-procedure called the Consensus Neural Network (ConsensusNet). Given a set of initial denoisers, ConsensusNet takes the initial estimates and generates a linear combination of the results. The combined estimate is then fed to a booster neural network to reduce the amount of method noise. ConsensusNet is a modular framework that allows any image denoiser to be used in the initial stage. Experimental results show that ConsensusNet can consistently improve denoising performance for both deterministic denoisers and neural network denoisers.


### Experiments
1. Noise-Level Mismatch (sigma = 10, 20, 30, 40, 50)
- REDNet
- DnCNN
2. Different Image Classes (classes = building, face, flower)
3. Different Denoiser Types (denoisers = DnCNN, FFDNet, BM3D, REDNet)

### Training Instructions
- Step1: [Train Neural Network](./train/1.single)
  ```
  python train.py --(options)
  ```
- Step2: [Combine](./combine)
  - Experiments 1
  ```
  python combine_noise.py --(options)
  ```
  - Experiments 2
  ```
  python combine_class.py --(options)
  ```
  - Experiments 3: 
    - Add your paths for MatConvNet, DnCNN, FFDNet and BM3D of combine_denoiser2.m
    ```
    run /home/matconvnet-1.0-beta25/matlab/vl_setupnn
    dncnn_path	= '...';
    bm3d_path	= '...';
    ffdnet_path	= '...';
    ```
    - Run codes
    ```
    python combine_denoiser1.py --(options)
    (matlab) combine_denoiser2.m
    ```
- Step3: [Train Booster](./train/2.booster)
  ```
  python train.py --(options)
  ```

### Testing Instructions
- Step1: Same as [Step2 in Training Instructions](./README#22)
- Step2: [Booster](./test)
  ```
  python test.py --(options)
  ```

### Requirements and Dependencies
- Cuda-8.0 & cuDNN v-5.1
- [Tensorflow 1.2.1](https://www.tensorflow.org/)
- [MatConvNet](http://www.vlfeat.org/matconvnet/) & MATLAB R2017a: for Experiments 1-b & 3
- [DnCNN](https://github.com/cszn/DnCNN): for Experiments 1-b & 3
- [FFDNet](https://github.com/cszn/FFDNet): for Experiments 3
