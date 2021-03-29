
import os,sys,re,pickle
from tqdm import tqdm

sys.path.append('/data/duongdb/StyleFlow')
os.chdir('/data/duongdb/StyleFlow')

from utils import * 

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g

# source /data/$USER/conda/etc/profile.d/conda.sh
# conda activate py37

# module load CUDA/11.0
# module load cuDNN/8.0.3/CUDA-11.0
# module load gcc/8.3.0

# cd /data/duongdb/StyleFlow

# ! follow https://github.com/RameenAbdal/StyleFlow/issues/14#issuecomment-765644589

# input_file_path = '/data/duongdb/WS_Aging02112021/aligned_images/'

# ---------------------------------------------------------------------------


input_file_path = sys.argv[1] # input folder 
output_file_path = sys.argv[2] # output folder
output_size = int ( sys.argv[3] ) # may be use 256 ? # ! they (StyleFlow) use default 

if not os.path.exists (output_file_path): 
  os.mkdir(output_file_path)


# ---------------------------------------------------------------------------


for f in tqdm ( os.listdir(input_file_path) ) : 
  Align_face_image(os.path.join(input_file_path,f), output_size=output_size, transform_size=4096, enable_padding=True, dest_file=os.path.join(output_file_path,f))



