#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:4 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=32g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=12

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada
# python dataset_tool.py create_from_images ~/datasets/metfaces ~/downloads/metfaces/images # ! don't need because used the same format as stylegan1/2

# python dataset_tool.py create_from_images /data/duongdb/NF1BeforeAfter01202021/Crop/SmallBigGanDataCopy2Tfrecord /data/duongdb/NF1BeforeAfter01202021/Crop/SmallBigGanDataCopy2 --resolution 512 # ! has to use 512, because the loaded ffhq1024 is taking 512 input

# ! from the help page The first command is optional; it will validate the arguments, print out the resulting training configuration, and exit. The second command will kick off the actual training.
# python3 train.py \
# --data=/data/duongdb/NF1BeforeAfter01202021/Crop/SmallBigGanDataCopy2Tfrecord \
# --mirror=1 \
# --aug=ada --target=0.7 \
# --augpipe=bgcfnc \
# --resume=ffhq1024 --snap=10 \
# --outdir=/data/duongdb/NF1BeforeAfter01202021/Crop/training-runs-stylegan2-ada \
# --dry-run

# ! doing training

cd /data/duongdb/stylegan2-ada
python3 train.py \
--data=/data/duongdb/NF1BeforeAfter01202021/Crop/SmallBigGanDataCopy2Tfrecord \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgcfnc \
--outdir=/data/duongdb/NF1BeforeAfter01202021/Crop/training-runs-stylegan2-ada \
--resume=/data/duongdb/NF1BeforeAfter01202021/Crop/training-runs-stylegan2-ada/00002-SmallBigGanDataCopy2Tfrecord-mirror-auto1-ada-target0.7-bgcfnc-resumeffhq1024/network-snapshot-001320.pkl

# --resume=ffhq1024 --snap=10 \