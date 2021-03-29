#!/bin/bash

# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=4



# ! cut out white space. 
# we can reuse same code we had for NF1 skin 
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
codepath=$datadir/HAM10000_dataset/EnhanceQuality
cd $codepath
for type in WSbefore WSafter WSintermediate # ! best to name so that inter. is rank last, we take average of first 2 labels
do 
  datapath=/data/duongdb/WS_Aging02112021/$type
  python3 CropWhiteSpaceCenter.py $datapath 0 png
  mkdir /data/duongdb/WS_Aging02112021/TrimWhiteSpaceImg
  mv $datapath/TrimWhiteSpaceNoBorder/*png /data/duongdb/WS_Aging02112021/TrimWhiteSpaceImg
done 
cd /data/duongdb/WS_Aging02112021/TrimWhiteSpaceImg


# ! align images into ffhq format
# this has to be done so we can greatly leverage transfer-ability of ffhq
resolution=512
datapath=/data/duongdb/WS_Aging02112021
cd $datadir/stylegan2-ada/WsData
python3 AlignImage.py $datapath/TrimWhiteSpaceImg $datapath/ImgAlignTfrecord$resolution $resolution

# ! align fairface data
resolution=256
datapath=/data/duongdb/FairFace/
cd $datadir/stylegan2-ada/WsData
python3 AlignImage.py $datapath/Subset1kRs1Align $datapath/Subset1kRs1Align$resolution $resolution # ! uses face align to remove img with 2 faces, then run this one. 


# ! make tfrecord data. 
cd /data/duongdb/stylegan2-ada
datapath=/data/duongdb/WS_Aging02112021
resolution=1024
python dataset_tool.py create_from_images_with_labels $datapath/ImgAlignTfrecord$resolution'Label3' $datapath/ImgAlignTfrecord$resolution --label_names 'WSbefore,WSafter,WSintermediate' --resolution $resolution 


# ! make tfrecord data. without alignment
cd /data/duongdb/stylegan2-ada
datapath=/data/duongdb/WS_Aging02112021
resolution=256
python dataset_tool.py create_from_images_with_labels $datapath/ImgUnalignTfrecord$resolution'Label3' $datapath/TrimWhiteSpaceImg --label_names 'WSbefore,WSafter,WSintermediate' --resolution $resolution 




