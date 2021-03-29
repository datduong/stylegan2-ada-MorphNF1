#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=3:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada

# ! doing training

# outputdir=/data/duongdb/WS_Aging02112021/
# # imagedata=$outputdir/ImgAlignTfrecord256wLabel
# resolution=256
# imagedata=$outputdir/ImgAlignTfrecord$resolution'Label3'

# cd /data/duongdb/stylegan2-ada
# python3 train_with_labels.py \
# --data=$imagedata \
# --gpus=1 \
# --mirror=1 \
# --aug=ada --target=0.7 \
# --augpipe=bgc \
# --metrics=fid100_full \
# --outdir=$outputdir/stylegan2-ada-Lab3-UnconLabEmb \
# --resume=ffhq256 

# --intermediate_state 

# ! best so far?
# network-snapshot-001011


#----------------------------------------------------------------------------
# ! generate images, using labels indexing
# ! let's try same random vector, but different label class

path='/data/duongdb/WS_Aging02112021/stylegan2-ada-Lab3-UnconLabEmb/00000-ImgAlignTfrecord256Label3-mirror-auto1-ada-target0.7-bgc-resumeffhq256'
model=$path/network-snapshot-001011.pkl ## ! the latest save is bad? 000768 is best
cd /data/duongdb/stylegan2-ada

truncationpsi=0.7 # @trunc=0.7 is recommended on their face dataset. 

# for class in 0 1 2 
# do 
# python3 generate.py --outdir=$path/SameZvecClass$class --trunc=$truncationpsi --seeds=0-100 --class=$class --network $model
# done 

# for mix_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do 
# python3 generate.py --outdir=$path/SameZvecClass0-1-$mix_ratio --trunc=$truncationpsi --seeds=0-100 --network $model --mix_label --mix_ratio $mix_ratio
# done 

# python3 concat_generated_img.py $path 
# cd $path 

# ! make data to train young old
data2trainClassifier=$path/'TrainLabelClassifier'
mkdir $data2trainClassifier

mkdir $data2trainClassifier/train
for class in 0 1 
do 
python3 generate.py --outdir=$data2trainClassifier/train/$class --trunc=$truncationpsi --seeds=0-1000 --class=$class --network $model
done 

mkdir $data2trainClassifier/dev
for class in 0 1 
do 
python3 generate.py --outdir=$data2trainClassifier/dev/$class --trunc=$truncationpsi --seeds=2000-2100 --class=$class --network $model
done 

mkdir $data2trainClassifier/test
for class in 0 1 
do 
python3 generate.py --outdir=$data2trainClassifier/test/$class --trunc=$truncationpsi --seeds=3000-3100 --class=$class --network $model
done 

