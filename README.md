
# What Does Rotation Prediction Tell Us about Classifier Accuracy under Varying Testing Environments?
## [[Paper]](https://weijiandeng.xyz/Rotation/files/ICML'21.pdf) [[ICML'21 Project]](https://weijiandeng.xyz/Rotation/)
![](https://weijiandeng.xyz/assets/images/projects/ICML_21.PNG)


## PyTorch Implementation

This repository contains:

- the PyTorch implementation of AutoEavl.
- the example on CIFAR-10 setup (use [imgaug](https://imgaug.readthedocs.io/en/latest/))
- linear regression

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Linux (tested on Ubuntu 16.04LTS)
* NVIDIA GPU + CUDA CuDNN (tested on GTX 2080 Ti)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (download and unzip to ```PROJECT_DIR/data/```)
* [CIFAR10.1](https://github.com/modestyachts/CIFAR-10.1) (download and unzip to ```PROJECT_DIR/data/CIFAR-10.1```)
* Please use PyTorch1.5 to avoid compilation errors (other versions should be good)
* You might need to change the file paths, and please be sure you change the corresponding paths in the codes as well     

## Getting started
0. Install dependencies 
    ```bash
   # Imgaug (or see https://imgaug.readthedocs.io/en/latest/source/installation.html)
    conda config --add channels conda-forge
    conda install imgaug
    ```
 1. Creat synthetic sets
    ```bash
    # By default it creates 500 synthetic sets
    python generate_synthetic_sets.py
    ```
 2. Learn classifier on CIFAR-10 (DenseNet-10-12)
    ```bash
    # Save as "PROJECT_DIR/DenseNet-40-12-ss/checkpoint.pth.tar"
    # Modified based on the wonderful github of https://github.com/andreasveit/densenet-pytorch
    python train.py --layers 40 --growth 12 --no-bottleneck --reduce 1.0
    ```
 3. Test classifier on synthetic sets
    ```bash
    # 1) Get "PROJECT_DIR/accuracy_cls_dense_aug.npy" file
    # 2) Get "PROJECT_DIR/accuracy_ss_dense_aug.npy" file
    # 3) You will see Rank correlation and Pearsons correlation
    # 4) The absolute error of linear regression is also shown
    python test_many.py --layers 40 --growth 12 --no-bottleneck --reduce 1.0
    ```
 4. Correlation study
    ```bash
    # You will see correlation.pdf;
    python analyze_correlation.py
        
## Citation
If you use the code in your research, please cite:
```bibtex
    @inproceedings{Deng:ICML2021,
      author    = {Weijian Deng and
                   Stephen Gould and
                   Liang Zheng},
      title     = {What Does Rotation Prediction Tell Us about Classifier Accuracy under Varying Testing Environments?},
      booktitle = {ICML},
      year      = {2021}
    }
```

## License
MIT
