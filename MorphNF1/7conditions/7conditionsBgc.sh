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
# for labelname in EverythingElse ML NF1 HMI IP MA TSC
# do 
#   scp /data/duongdb/SkinConditionImages01112021/Crop/$labelname/TrimWhiteSpaceNoBorder/*jpg /data/duongdb/SkinConditionImages01112021/CropImgJpg/
# done


cd /data/duongdb/stylegan2-ada
# # python dataset_tool.py create_from_images ~/datasets/metfaces ~/downloads/metfaces/images # ! don't need because used the same format as stylegan1/2

# # python dataset_tool.py create_from_images /data/duongdb/SkinConditionImages01112021/Crop/SmallBigGanDataCopy2Tfrecord /data/duongdb/SkinConditionImages01112021/Crop/SmallBigGanDataCopy2 --resolution 512 # ! has to use 512, because the loaded ffhq1024 is taking 512 input

# resolution=512
# python dataset_tool.py create_from_images_with_labels /data/duongdb/SkinConditionImages01112021/Crop/Tfrecord$resolution'wLabel' /data/duongdb/SkinConditionImages01112021/CropImgJpg/ --label_names 'EverythingElse,ML,NF1,HMI,IP,MA,TSC' --resolution $resolution # ! has to use 512 if load ffhq1024 which is taking 512 input


# ! from the help page The first command is optional; it will validate the arguments, print out the resulting training configuration, and exit. The second command will kick off the actual training.
# python3 train.py \
# --data=/data/duongdb/SkinConditionImages01112021/Crop/SmallBigGanDataCopy2Tfrecord \
# --mirror=1 \
# --aug=ada --target=0.7 \
# --augpipe=bgcfnc \
# --resume=ffhq1024 --snap=10 \
# --outdir=/data/duongdb/SkinConditionImages01112021/Crop/training-runs-stylegan2-ada \
# --dry-run

# ! doing training
# ! use bgc or bgcfnc
cd /data/duongdb/stylegan2-ada
python3 train_with_labels.py \
--data=/data/duongdb/SkinConditionImages01112021/Crop/Tfrecord256wLabel \
--gpus=2 \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--metrics=fid200_full \
--outdir=/data/duongdb/SkinConditionImages01112021/Crop/training-stylegan2-ada-Lab3-BaseLabEmb \
--resume=ffhq256 # ! not use intermediate average


# #----------------------------------------------------------------------------
# # ! generate images, using labels indexing
# # ! let's try same random vector, but different label class

path='/data/duongdb/SkinConditionImages01112021/Crop/training-stylegan2-ada-Lab3-BaseLabEmb/00000-Tfrecord256wLabel-mirror-auto2-ada-target0.7-bgc-resumeffhq256'
model=$path/network-snapshot-000768.pkl
cd /data/duongdb/stylegan2-ada

for class in 0 1 2 3 4 5 6
do 
python3 generate.py --outdir=$path/SameZvecClass$class --trunc=0.7 --seeds=0-25 --class=$class --network $model
done 

# for mix_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do 
# python3 generate.py --outdir=$path/SameZvecClass0-1-$mix_ratio --trunc=0.7 --seeds=0-100 --network $model --mix_label --mix_ratio $mix_ratio
# done 

# python3 concat_generated_img.py $path 
# cd $path 

# #----------------------------------------------------------------------------
# # ! let's do interpolation in W space like what people usually do with face data

# maindir='/data/duongdb/SkinConditionImages01112021/Crop'
# expterimenttype='training-stylegan2-ada-Lab3-BaseLabEmb'
# model_path=$maindir/$expterimenttype/'00000-Tfrecord256wLabel-mirror-auto2-ada-target0.7-bgc-resumeffhq256/network-snapshot-000768.pkl'

# for imgtype in Before After Inter
# do 
#   cd /data/duongdb/stylegan2-ada
#   originalselect=$maindir/ImgRunProjectorInput$imgtype
#   outdir=$maindir/$expterimenttype/ImgRunTfrecord$imgtype
#   python3 dataset_tool.py create_from_images_with_labels $outdir $originalselect --resolution 256 --shuffle 0 --label_names 'NF1Before,NF1After,NF1Inter'
# done 

# # ! project on real images
# for imgtype in Before After Inter
# do 
# cd /data/duongdb/stylegan2-ada 
# tot_aligned_imgs=4 # ! change this number
# python3 run_projector.py project-real-images --network=$model_path --data-dir=$maindir/$expterimenttype/ImgRunTfrecord$imgtype --num-images=$tot_aligned_imgs --num-snapshots 4
# done

# # ! run interpolation W space
# cd /data/duongdb/stylegan2-ada/Interpolate 
# recon_img=/data/duongdb/stylegan2-ada/results
# python3 run_interpolate.py $maindir/$expterimenttype/'00000-Tfrecord256wLabel-mirror-auto2-ada-target0.7-bgc-resumeffhq256/' 'network-snapshot-000768.pkl' $recon_img/'00013-project-real-images' $recon_img/'00014-project-real-images' 

# # ! interpolate label space. notice something is strange. the images reconstructed is not the same as @generate.py ???
# cd /data/duongdb/stylegan2-ada/Interpolate 
# recon_img=/data/duongdb/stylegan2-ada/results
# python3 run_interpolate_label.py $maindir/$expterimenttype/'00000-Tfrecord256wLabel-mirror-auto2-ada-target0.7-bgc-resumeffhq256/' 'network-snapshot-000768.pkl' 
# cd $maindir/$expterimenttype/

