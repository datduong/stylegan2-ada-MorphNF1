#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada

# ! doing training

outputdir=/data/duongdb/WS_Aging02112021/
# imagedata=$outputdir/ImgAlignTfrecord256wLabel
resolution=256
imagedata=$outputdir/ImgAlignTfrecord$resolution'Label3'

cd /data/duongdb/stylegan2-ada
python3 train_with_labels.py \
--data=$imagedata \
--gpus=1 \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--metrics=fid100_full \
--outdir=$outputdir/stylegan2-ada-Lab3-aveLabEmb \
--resume=ffhq256 --intermediate_state 

# ! best so far?
# network-snapshot-001264


#----------------------------------------------------------------------------
# ! generate images, using labels indexing
# ! let's try same random vector, but different label class

path='/data/duongdb/WS_Aging02112021/stylegan2-ada-Lab3-aveLabEmb/00000-ImgAlignTfrecord256Label3-mirror-auto1-ada-target0.7-bgc-resumeffhq256'
model=$path/network-snapshot-001264.pkl ## ! the latest save is bad? 000768 is best
cd /data/duongdb/stylegan2-ada

truncationpsi=0.7 # @trunc=0.7 is recommended on their face dataset. 

for class in 0 1 2 
do 
python3 generate.py --outdir=$path/SameZvecClass$class --trunc=$truncationpsi --seeds=0-100 --class=$class --network $model
done 

for mix_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do 
python3 generate.py --outdir=$path/SameZvecClass0-1-$mix_ratio --trunc=$truncationpsi --seeds=0-100 --network $model --mix_label --mix_ratio $mix_ratio
done 

python3 concat_generated_img.py $path 
cd $path 


#----------------------------------------------------------------------------
# ! try on real images

# # sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g
# source /data/$USER/conda/etc/profile.d/conda.sh
# conda activate py37

# module load CUDA/11.0
# module load cuDNN/8.0.3/CUDA-11.0
# module load gcc/8.3.0

# cd /data/duongdb/stylegan2encoder
# imgdir=/data/duongdb/WS_Aging02112021
# resolution=256

# path='/data/duongdb/WS_Aging02112021/stylegan2-ada-Lab3-aveLabEmb/00000-ImgAlignTfrecord256Label3-mirror-auto1-ada-target0.7-bgc-resumeffhq256'
# model=$path/network-snapshot-001264.pkl 

# # ! load our model 
# # ! error....
# python project_images.py $imgdir/ImgAlignTfrecord$resolution'Label3'/ $imgdir/ImgAlignTfrecord$resolution'Label3ReconImg/' --vgg16-pkl /data/duongdb/vgg16_zhang_perceptual.pkl --network-pkl $model 



