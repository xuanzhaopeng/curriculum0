# Install the verl into curriculum0 environment
cd /workspace

VERL_DIR="/workspace/verl"
VERL_TAG="v0.7.0"

if [ ! -d "$VERL_DIR" ]; then
    git clone --branch $VERL_TAG https://github.com/volcengine/verl.git
else
    cd $VERL_DIR
    git checkout $VERL_TAG
fi

cd /workspace/verl
pip install --no-deps -e .