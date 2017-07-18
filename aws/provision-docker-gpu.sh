#!/usr/bin/env bash
# This assumes ubuntu 16.04
sudo df -h
sudo apt-get update
sudo apt-get -y install tmux build-essential gcc g++ make binutils
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
# List the drives for my viewing pleasure
# Install Docker
sudo apt-get -y remove docker docker-engine docker.io
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get -y install docker-ce
echo "Test docker"
sudo docker ps

# Install NVIDIA
NVIDIA_VERSION=8.0.61-1
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${NVIDIA_VERSION}_amd64.deb" -O "cuda-repo-ubuntu1604_${NVIDIA_VERSION}_amd64.deb"

sudo dpkg -i cuda-repo-ubuntu1604_${NVIDIA_VERSION}_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo modprobe nvidia
echo "Test nvidia-smi"
nvidia-smi

# Install NVIDIA-Docker
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Test nvidia-smi in docker
echo "Test nvidia-smi in docker"
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

# Install Anaconda + other things
ANACONDA_VERSION=Anaconda2-4.4.0
mkdir setup-py
cd setup-py
wget "https://repo.continuum.io/archive/${ANACONDA_VERSION}-Linux-x86_64.sh" -O "${ANACONDA_VERSION}-Linux-x86_64.sh"
bash "${ANACONDA_VERSION}-Linux-x86_64.sh" -b

echo "export PATH=\"$HOME/anaconda2/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda2/bin:$PATH"
conda install -y bcolz
conda upgrade -y --all

# install and configure theano
pip install theano
echo "[global]
device = cuda0
floatX = float32
[cuda]
root = /usr/local/cuda" > ~/.theanorc

# install and configure keras
pip install keras
mkdir ~/.keras
echo '{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}' > ~/.keras/keras.json

# install tensorflow
pip install tensorflow-gpu
# install pygpu
conda install pygpu
# install cudnn libraries
wget "http://files.fast.ai/files/cudnn.tgz" -O "cudnn.tgz"
tar -zxf cudnn.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/

# Some cleanup
sudo apt-get clean
sudo apt-get autoclean

# Test libs work
echo "from keras import backend as K" | python
echo "import tensorflow as tf" | python
echo "import tensorflow;tensorflow.train.Server.create_local_server()" | python

# Install scikit-learn
conda install scikit-learn
# Configure Jupyter
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py