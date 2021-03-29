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

# ! scp over. 
# for labelname in NF1After NF1Before NF1Inter 
# do 
#   scp /data/duongdb/NF1BeforeAfterInter03182021/Crop/$labelname/TrimWhiteSpaceNoBorder/*jpg /data/duongdb/NF1BeforeAfterInter03182021/CropImgJpg/
# done

resolution=256

cd /data/duongdb/stylegan2-ada
# python dataset_tool.py create_from_images ~/datasets/metfaces ~/downloads/metfaces/images # ! don't need because used the same format as stylegan1/2

python dataset_tool.py create_from_images /data/duongdb/NF1BeforeAfterInter03182021/Crop/Tfrecord$resolution'NoLabel' /data/duongdb/NF1BeforeAfterInter03182021/CropImgJpg/ --resolution $resolution # ! resolution depends on the pre-trained model 

# ! doing training
cd /data/duongdb/stylegan2-ada
python3 train.py \
--data=/data/duongdb/NF1BeforeAfterInter03182021/Crop/Tfrecord$resolution'NoLabel' \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--outdir=/data/duongdb/NF1BeforeAfterInter03182021/Crop/training-stylegan2-ada-Lab3-NoLabEmb \
--resume=ffhq$resolution 

# --snap=10 

# ! generate images, using labels indexing if needed.
# cd /data/duongdb/stylegan2-ada
# python generate.py --outdir=/data/duongdb/NF1BeforeAfterInter03182021/Crop/training-runs-stylegan2-ada --trunc=1 --seeds=0-20 --class=0 --network /data/duongdb/NF1BeforeAfterInter03182021/Crop/training-runs-stylegan2-ada/00014-Tfrecord256wLabel-mirror-auto1-ada-target0.7-bgcfnc/network-snapshot-006320.pkl

# ! 
