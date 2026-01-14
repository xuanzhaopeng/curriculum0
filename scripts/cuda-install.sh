# cuda 12.8 installation

mkdir -p /workspace/.cuda
cd /workspace/.cuda

CUDA_DEB="cuda-repo-ubuntu2204-12-8-local_12.8.1-570.124.06-1_amd64.deb"

if [ ! -f "$CUDA_DEB" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/$CUDA_DEB
fi

dpkg -i $CUDA_DEB
cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-8 
update-alternatives --set cuda /usr/local/cuda-12.8
# check the cuda version
nvcc --version
