CONDA_ROOT="/workspace/.miniconda3"

if [ ! -d "$CONDA_ROOT" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_ROOT"
    rm Miniconda3-latest-Linux-x86_64.sh
fi

$CONDA_ROOT/bin/conda init bash && source ~/.bashrc
