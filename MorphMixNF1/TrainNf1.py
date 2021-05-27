import re,sys,os,pickle

script="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! set folder path

workdir=WORKDIR
our_data_path=$workdir/Example


# ! make data into tfrecords (we already gave you the tfrecords, so you don't need to run it)

# cd $workdir
# python dataset_tool.py create_from_images_with_labels $our_data_path/EarlyInterLateNF1/Tfrecord256Label $our_data_path/EarlyInterLateNF1/CropImgJpg --label_names 'NF1Before,NF1After,NF1Inter' --resolution 256 


# ! training

cd $workdir
python3 train_with_labels.py \
--data=$our_data_path/EarlyInterLateNF1/Tfrecord256Label \
--gpus=2 \
--mirror=1 \
--aug=ada --target=0.7 \
--augpipe=bgc \
--metrics=kid50k_full,fid50k_full \
--outdir=$our_data_path/EarlyInterLateNF1Model \
--resume=ffhq256 \
--cfg=CONFIG \
--intermediate_state 

"""


import time
from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

workdir = '/data/duongdb/DeployOnline/stylegan2-ada-MorphNF1'
os.chdir(workdir)
counter = 1

for CONFIG in ['paper256'] : 
  newscript = re.sub('CONFIG',str(CONFIG),script)
  newscript = re.sub('WORKDIR',str(workdir),newscript)
  fname = 'script'+str(counter+1)+date_time+'.sh'
  fout = open(fname,'w')
  fout.write(newscript)
  fout.close()
  counter = counter + 1
  print ('make script {} at {}'.format(fname,workdir))
  # time.sleep(60)
  # os.system('sbatch --partition=gpu --time=20:00:00 --gres=gpu:v100x:2 --mem=16g --cpus-per-task=8 '+fname)



