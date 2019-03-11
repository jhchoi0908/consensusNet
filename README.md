## [Optimal Combination of Image Denoisers](https://arxiv.org/abs/1711.06712) (ConsensusNet)

### Abstract

Given a set of image denoisers, each having a different denoising capability, is there a provably optimal way of combining these denoisers to produce an overall better result? An answer to this question is fundamental to designing an ensemble of weak estimators for complex scenes. In this paper, we present an optimal combination scheme by leveraging deep neural networks and convex optimization. The proposed framework, called the Consensus Neural Network (CsNet), introduces three new concepts in image denoising: (1) A provably optimal procedure to combine the denoised outputs via convex optimization; (2) A deep neural network to estimate the mean squared error (MSE) of denoised images without needing the ground truths; (3) An image boosting procedure using a deep neural network to improve contrast and to recover lost details of the combined images. Experimental results show that CsNet can consistently improve denoising performance for both deterministic and neural network denoisers.


### Experiments
1. Different Noise Level (sigma = 10, 20, 30, 40, 50)
2. Different Image Classes (classes = building, face, flower)
3. Different Denoiser Types (denoisers = DnCNN, FFDNet, BM3D, REDNet)


### Testing Instructions (Please modify the folder for dataset)
- Experiments 1 and 2
  ```
  python test.py --(options)
  ```
- Experiment 3
  ```
  matlab denoisers.m
  python rednet.py --(options)
  python test.py --(options)
  ```

### Requirements and Dependencies
- Cuda-8.0 & cuDNN v-5.1
- [Tensorflow](https://www.tensorflow.org/) 1.2 or above
- [cvxpy](https://www.cvxpy.org/)
- [MatConvNet](http://www.vlfeat.org/matconvnet/) & MATLAB R2017a: for Different Denoiser Types
- [DnCNN](https://github.com/cszn/DnCNN): for Different Denoiser Types 
- [FFDNet](https://github.com/cszn/FFDNet): for Different Denoiser Types


### Citations
```
@article{choi2019,
  title={Optimal Combination of Image Denoisers},
  author={Choi, Joon Hee and Elgendy, Omar A. and Chan, Stanley H.},
  journal={IEEE Transactions on Image Processing},
  year={2019},
  volume={}, 
  number={}, 
  pages={},
  doi={10.1109/TIP.2019.2903321}
}
```
