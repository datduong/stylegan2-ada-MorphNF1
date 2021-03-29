import re,sys,os,pickle

script="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada

# ! make data into tfrecords
# python dataset_tool.py create_from_images_with_labels /data/duongdb/NF1BeforeAfterInter03182021/Crop/Tfrecord256Label /data/duongdb/NF1BeforeAfterInter03182021/CropImgJpg/ --label_names 'NF1Before,NF1After,NF1Inter' --resolution 256 

# ! training

cd /data/duongdb/stylegan2-ada
python3 train_with_labels.py \
--data=/data/duongdb/NF1BeforeAfterInter03182021/Crop/Tfrecord256Label \
--gpus=2 \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--metrics=kid50k_full,fid50k_full \
--outdir=/data/duongdb/NF1BeforeAfterInter03182021/Crop/1HotAveEmb \
--resume=ffhq256 \
--cfg=CONFIG \
--intermediate_state 

"""

# paper512 auto paper256
# bgcfnc

import time
from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

os.chdir('/data/duongdb/stylegan2-ada')
counter = 1

for CONFIG in ['paper256'] : # ! 1-->use label vec or 0-->use direct 1-hot
  newscript = re.sub('CONFIG',str(CONFIG),script)
  fname = 'script'+str(counter+1)+date_time+'.sh'
  fout = open(fname,'w')
  fout.write(newscript)
  fout.close()
  counter = counter + 1
  time.sleep(60)
  os.system('sbatch --partition=gpu --time=20:00:00 --gres=gpu:v100x:2 --mem=16g --cpus-per-task=8 '+fname)



