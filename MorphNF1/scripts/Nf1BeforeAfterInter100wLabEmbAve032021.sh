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

# python dataset_tool.py create_from_images_with_labels /data/duongdb/NF1BeforeAfterInter03182021/Crop/Tfrecord256Label /data/duongdb/NF1BeforeAfterInter03182021/CropImgJpg/ --label_names 'NF1Before,NF1After,NF1Inter' --resolution 256 

# ! training

cd /data/duongdb/stylegan2-ada
python3 train_with_labels.py \
--data=/data/duongdb/NF1BeforeAfterInter03182021/Crop/Tfrecord256Label \
--gpus=2 \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--metrics=fid100_full \
--outdir=/data/duongdb/NF1BeforeAfterInter03182021/Crop/1HotAveEmb \
--resume=ffhq256 \
--cfg=paper512 \
--intermediate_state 


# # --snap=10 


#----------------------------------------------------------------------------
# ! generate images, using labels indexing
# ! let's try same random vector, but different label class

path='/data/duongdb/NF1BeforeAfterInter03182021/Crop/1HotAveEmb/00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256'
model=$path/network-snapshot-000768.pkl ## ! the latest save is bad? 000768 is best
cd /data/duongdb/stylegan2-ada

truncationpsi=0.6 # @trunc=0.7 is recommended on their face dataset. 

for class in 0 1 2 
do 
python3 generate.py --outdir=$path/SameZvecClass$class --trunc=$truncationpsi --seeds=0-200 --class=$class --network $model
done 

for mix_ratio in 0 0.25 .5 0.75 1
do 
python3 generate.py --outdir=$path/SameZvecMix$mix_ratio --trunc=$truncationpsi --seeds=0-200 --network $model --mix_ratio $mix_ratio --class 0 --class_next 1
done 

cd /data/duongdb/stylegan2-ada
python3 concat_generated_img.py $path '0 0.25 .5 0.75 1'
cd $path 

# #----------------------------------------------------------------------------
# # ! let's do interpolation in W space like what people usually do with face data

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

maindir='/data/duongdb/NF1BeforeAfterInter03182021/Crop'
expterimenttype='training-stylegan2-ada-Lab3-aveLabEmb'
model_path=$maindir/$expterimenttype/'00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256/network-snapshot-000768.pkl'

for imgtype in Before After # Inter
do 
  cd /data/duongdb/stylegan2-ada
  originalselect=$maindir/ImgRunProjectorInput$imgtype
  outdir=$maindir/$expterimenttype/ImgRunTfrecord$imgtype
  rm -rf $outdir
  mkdir $outdir
  python3 dataset_tool.py create_from_images_with_labels $outdir $originalselect --resolution 256 --shuffle 0 --label_names 'NF1Before,NF1After,NF1Inter'
done 

# ! project on real images
recon_img=$maindir/$expterimenttype/'00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256/'

rm -rf $recon_img/*project*real* 

for imgtype in Before After # Inter
do 
cd /data/duongdb/stylegan2-ada 
tot_aligned_imgs=8 # ! change this number
python3 run_projector.py project-real-images --network=$model_path --data-dir=$maindir/$expterimenttype/ImgRunTfrecord$imgtype --num-images=$tot_aligned_imgs --num-snapshots 4 --result-dir $recon_img
done

# ! run interpolation W space
truncationpsi=0.7 # @trunc=0.7 is recommended on their face dataset. but we were using 0.5
recon_img=$maindir/$expterimenttype/'00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256/'

cd /data/duongdb/stylegan2-ada/Interpolate 
python3 run_interpolate.py $maindir/$expterimenttype/'00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256/' 'network-snapshot-000768.pkl' $recon_img/'00000-project-real-images' $recon_img/'00001-project-real-images' $truncationpsi

cd $maindir/$expterimenttype/'00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256/' 
mkdir network-snapshot-000768
mv $recon_img/'00000-project-real-images' $recon_img/'00001-project-real-images' Same* Interpolate Gen*Interpolate network-snapshot-000768




# # ! interpolate label space. notice something is strange. the images reconstructed is not the same as @generate.py ???
# cd /data/duongdb/stylegan2-ada/Interpolate 
# recon_img=/data/duongdb/stylegan2-ada/results
# python3 run_interpolate_label.py $maindir/$expterimenttype/'00001-Tfrecord256Label-mirror-auto2-ada-target0.7-bgc-resumeffhq256/' 'network-snapshot-000768.pkl' 
# cd $maindir/$expterimenttype/
