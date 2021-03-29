import re,sys,os,pickle 

BASE = """#!/bin/bash

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
cd /data/duongdb/stylegan2-ada

resolution=256

# for fold in 0 1 2 3 4 
# do 
# python dataset_tool.py create_from_images_with_labels /data/duongdb/SkinConditionImages01112021/Crop/GanFold$fold'Tfrecord'$resolution'wLabel' /data/duongdb/SkinConditionImages01112021/Crop/GanFold$fold/ --label_names 'EverythingElse,ML,NF1,HMI,IP,MA,TSC' --resolution $resolution # ! has to use 512 if load ffhq1024 which is taking 512 input
# done 

# ! doing training
# ! use bgc or bgcfnc
cd /data/duongdb/stylegan2-ada
python3 train_with_labels.py \
--data=/data/duongdb/SkinConditionImages01112021/Crop/GanFoldFOLD'Tfrecord'$resolution'wLabel'  \
--gpus=2 \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--metrics=fid200_full \
--outdir=/data/duongdb/SkinConditionImages01112021/Crop/stylegan2-ada-BaseLabEmb-FOLD \
--resume=ffhq256 # ! not use intermediate average

# #----------------------------------------------------------------------------
# ! generate images, using labels indexing
# ! let's try same random vector, but different label class


path='/data/duongdb/SkinConditionImages01112021/Crop/stylegan2-ada-BaseLabEmb-0/00000-GanFold0Tfrecord256wLabel-mirror-auto2-ada-target0.7-bgc-resumeffhq256'
model=$path/network-snapshot-001792.pkl # ! will be different for each fold. maybe some folds are too hard?
cd /data/duongdb/stylegan2-ada

for class in 0 1 2 3 4 5 6
do 
  if [ $class = 0 ] ;
  then
  python3 generate.py --outdir=$path/RandZClass$class --trunc=0.7 --seeds=0-99 --class=$class --network $model
  fi  

  if [ $class = 1 ] ;
  then
  python3 generate.py --outdir=$path/RandZClass$class --trunc=0.7 --seeds=100-199 --class=$class --network $model
  fi  

  if [ $class = 2 ] ;
  then
  python3 generate.py --outdir=$path/RandZClass$class --trunc=0.7 --seeds=200-299 --class=$class --network $model
  fi  

  if [ $class = 3 ] ;
  then
  python3 generate.py --outdir=$path/RandZClass$class --trunc=0.7 --seeds=300-399 --class=$class --network $model
  fi  

  if [ $class = 4 ] ;
  then
  python3 generate.py --outdir=$path/RandZClass$class --trunc=0.7 --seeds=400-499 --class=$class --network $model
  fi  

  if [ $class = 5 ] ;
  then
  python3 generate.py --outdir=$path/RandZClass$class --trunc=0.7 --seeds=500-599 --class=$class --network $model
  fi  

  if [ $class = 6 ] ;
  then
  python3 generate.py --outdir=$path/RandZClass$class --trunc=0.7 --seeds=600-699 --class=$class --network $model
  fi  

done 


"""

os.chdir('/data/duongdb/SkinConditionImages01112021/Crop/')
for FOLD in range (0,5): 
  script = re.sub ('FOLD',str(FOLD), BASE)
  fout = open('script'+str(FOLD)+'.sh','w')
  fout.write( script )
  fout.close() 
  os.system ( 'sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 ' + 'script'+str(FOLD)+'.sh' )


