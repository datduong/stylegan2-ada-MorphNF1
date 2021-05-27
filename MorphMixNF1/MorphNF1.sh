#!/bin/bash

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! need to train model first !!

#----------------------------------------------------------------------------

# ! generate images, using labels indexing
# ! let's try same random vector, but different label class

workdir=/data/duongdb/DeployOnline/stylegan2-ada-MorphNF1
our_data_path=$workdir/Example

path=$our_data_path/EarlyInterLateNF1Model/00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256
model=$path/network-snapshot-000768.pkl ## ! load model

cd $workdir

truncationpsi=0.6 # @trunc=0.7 is recommended on their face dataset. 

for class in 0 1 2 
do 
python3 generate.py --outdir=$path/SameZvecClass$class --trunc=$truncationpsi --seeds=0-200 --class=$class --network $model
done 

for mix_ratio in 0 0.25 .5 0.75 1
do 
python3 generate.py --outdir=$path/SameZvecMix$mix_ratio --trunc=$truncationpsi --seeds=0-200 --network $model --mix_ratio $mix_ratio --class 0 --class_next 1
done 

cd $workdir
python3 concat_generated_img.py $path '0 0.25 .5 0.75 1'
cd $path 

