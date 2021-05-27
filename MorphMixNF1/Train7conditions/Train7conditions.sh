#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 

# ! set folder path

workdir=/data/duongdb/DeployOnline/stylegan2-ada-MorphNF1
our_data_path=$workdir/Example

# ! scp data. 
# for labelname in EverythingElse ML NF1 HMI IP MA TSC
# do 
#   scp $our_data_path/$labelname/TrimWhiteSpaceNoBorder/*jpg $our_data_path/CropImgJpg 
# done

# ! make tfrecords
# cd $workdir
# resolution=512
# python dataset_tool.py create_from_images_with_labels $our_data_path/Tfrecord$resolution'wLabel' $our_data_path/CropImgJpg --label_names 'EverythingElse,ML,NF1,HMI,IP,MA,TSC' --resolution $resolution 

# ! training
# ! use bgc or bgcfnc ?
cd $workdir
python3 train_with_labels.py \
--data=$our_data_path/Tfrecord256wLabel \
--gpus=2 \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--metrics=fid200_full \
--outdir=$our_data_path/Model7Conditions \
--resume=ffhq256 

# ! generate images, using labels indexing
# ! let's try same random vector, but different label class

path=$our_data_path/Model7Conditions/00000-Tfrecord256wLabel-mirror-auto2-ada-target0.7-bgc-resumeffhq256
model=$path/network-snapshot-000768.pkl

cd $workdir
for class in 0 1 2 3 4 5 6
do 
python3 generate.py --outdir=$path/SameZvecClass$class --trunc=0.7 --seeds=0-100 --class=$class --network $model
done 

# ! style-mixing

cd $workdir
class=5
python3 style_mixing.py --outdir=$path/StyleMix$class --trunc=0.7 --rows=821,458,293 --cols=100,75,1500 --class=$class --network $model
cd $path/StyleMix$class

