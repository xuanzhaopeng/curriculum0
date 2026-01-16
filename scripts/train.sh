echo "Start training curriculum"

CUDA_VISIBLE_DEVICES=0 python -m curriculum.train

echo "curriculum agent training finished"
