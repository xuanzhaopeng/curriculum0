

mkdir -p /workspace/.cudnn
cd /workspace/.cudnn
wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn-cuda-12
