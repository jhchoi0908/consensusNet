# Consensus Neural Network (ConsensusNet)

## [Optimal Combination of Image Denoisers](https://arxiv.org/abs/1711.06712)

### Abstract

Given a set of image denoisers, each having a different denoising capability, is there a provably optimal way of combining these denoisers to produce an overall better result? An answer to this question is fundamental to designing an ensemble of weak estimators for complex scenes. In this paper, we present an optimal combination scheme by leveraging deep neural networks and convex optimization. The proposed framework, called the Consensus Neural Network (CsNet), introduces three new concepts in image denoising: (1) A provably optimal procedure to combine the denoised outputs via convex optimization; (2) A deep neural network to estimate the mean squared error (MSE) of denoised images without needing the ground truths; (3) An image boosting procedure using a deep neural network to improve contrast and to recover lost details of the combined images. Experimental results show that CsNet can consistently improve denoising performance for both deterministic and neural network denoisers.


### Experiments
1. Different Noise Level (sigma = 10, 20, 30, 40, 50)
2. Different Image Classes (classes = building, face, flower)
3. Different Denoiser Types (denoisers = DnCNN, FFDNet, BM3D, REDNet)


### Testing Instructions
- Experiments 1 and 3
  ```
  python test.py --(options)
  ```
- Experiment 2
  ```
  matlab denoisers.m
  python rednet.py --(options)
  python test.py --(options)
  ```

### Requirements and Dependencies
- Cuda-8.0 & cuDNN v-5.1
- [Tensorflow 1.2.1](https://www.tensorflow.org/)
- [MatConvNet](http://www.vlfeat.org/matconvnet/) & MATLAB R2017a: for Different Denoiser Types
- [DnCNN](https://github.com/cszn/DnCNN): for Different Denoiser Types 
- [FFDNet](https://github.com/cszn/FFDNet): for Different Denoiser Types
