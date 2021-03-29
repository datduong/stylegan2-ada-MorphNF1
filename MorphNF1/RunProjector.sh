#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:4 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada

#----------------------------------------------------------------------------

# ! copy files over ?? which real images are chosen to be projected
maindir='/data/duongdb/NF1BeforeAfterInter02012021/Crop'
expterimenttype='training-stylegan2-ada-Lab3-NoLabEmb'

# model_path=$maindir/$expterimenttype/'00000-Tfrecord256NoLabel-mirror-auto1-ada-target0.7-bgc-resumeffhq256/network-snapshot-000758.pkl'

# ! scp image and convert them into tfrecords
# after 61 64 77 81 71 3 7 13
# before 18 21 52 59 86 72 15 34
# inter 18 19 24 29

for imgtype in After # Before After Inter
do
  mkdir $maindir/ImgRunProjectorInput$imgtype
  for num in 61 64 77 81 71 3 7 13 # 2 3 4 5 6 7 8 9 10 11
  do 
  scp /data/duongdb/NF1BeforeAfterInter02012021/Crop/NF1$imgtype/TrimWhiteSpaceNoBorder/NF1$imgtype'Slide'$num.jpg $maindir/ImgRunProjectorInput$imgtype
  done 
done

for imgtype in Before After # Inter
do 
  cd /data/duongdb/stylegan2-ada
  originalselect=$maindir/ImgRunProjectorInput$imgtype
  outdir=$maindir/$expterimenttype/ImgRunTfrecord$imgtype
  rm -rf $outdir
  mkdir $outdir
  python3 dataset_tool.py create_from_images $outdir $originalselect --resolution 256 --shuffle 0
done 

#----------------------------------------------------------------------------

# ! project on real images
for imgtype in Before After Inter
do 
cd /data/duongdb/stylegan2-ada 
tot_aligned_imgs=4 # ! change this number
python3 run_projector.py project-real-images --network=$model_path --data-dir=$maindir/$expterimenttype/ImgRunTfrecord$imgtype --num-images=$tot_aligned_imgs --num-snapshots 4
done

#----------------------------------------------------------------------------


