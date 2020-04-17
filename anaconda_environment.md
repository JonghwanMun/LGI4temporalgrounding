## Environment
We test the project on Ubuntu 16.04(Linux), Anaconda3 (v5.2 with python3.6).

### 1. Setting Anaconda environment
After installing [Anaconda](https://www.continuum.io/downloads), it is nice to create a [conda environment](http://conda.pydata.org/docs/using/envs.html)
so you do not destroy your main installation in case you make a mistake somewhere:
```bash
conda create --name tg python=3.6
```
Now you can switch to the new environment in your terminal by running the following (on Linux terminal):
```bash
source activate tg 
```

### 2. Required Packages
#### PyTorch installation

Refer to [PyTorch](http://pytorch.org/).
The tested versions are v1.1.0 (py3.6_cuda9.0.176_cudnn7.5.1_0) and v0.3.0 for pytorch and torchvision, respectively.
PC environment is Ubuntu 16.04 (Linux), Anaconda3 (v5.2 with python3.6).
```bash
conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
```

#### TensorFlow for tensorboard (python 3.6)
We install CPU-version of TensorFlow since we use only tensorboard.
```bash
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
```

#### Python packages
This project requires several Python packages to be installed.<br />
You can install the packages by typing:
```bash
conda install -y nb_conda numpy scipy jupyter matplotlib pillow nltk tqdm pyyaml scikit-image scikit-learn h5py
conda install -y -c conda-forge coloredlogs
pip install moviepy
```

