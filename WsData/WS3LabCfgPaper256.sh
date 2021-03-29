#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada

# ! doing training

outputdir=/data/duongdb/WS_Aging02112021/
resolution=256
imagedata=$outputdir/ImgAlignTfrecord$resolution'Label3'

cd /data/duongdb/stylegan2-ada
python3 train_with_labels.py \
--data=$imagedata \
--gpus=2 \
--aug=ada --target=0.7 \
--metrics=fid100_full \
--outdir=$outputdir/stylegan2-ada-Lab3-aveLabEmb \
--resume=ffhq256 --intermediate_state \
--cfg=paper256 --aug=ada --cmethod=bcr # ! same config as 10k subset of FFHQ with ADA and bCR
