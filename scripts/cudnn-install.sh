

mkdir -p /workspace/.cudnn
cd /workspace/.cudnn


CUDNN_DEB="cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb"

if [ ! -f "$CUDNN_DEB" ]; then
    wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/$CUDNN_DEB
fi

dpkg -i $CUDNN_DEB
cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn-cuda-12
