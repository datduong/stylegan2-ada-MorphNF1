

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=6g --cpus-per-task=6

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2encoder

imgdir=/data/duongdb/WS_Aging02112021

# ! need to create tfrecord ? 

# ! load the default ffhq
# python project_images.py $imgdir/aligned_images/ $imgdir/generated_images/ --vgg16-pkl /data/duongdb/vgg16_zhang_perceptual.pkl

# # ! load our model 
# python project_images.py $imgdir/aligned_images/ $imgdir/generated_images/ --vgg16-pkl /data/duongdb/vgg16_zhang_perceptual.pkl

# ! load our model 
path='/data/duongdb/WS_Aging02112021/stylegan2-ada-Lab3-UnconLabEmb/00000-ImgAlignTfrecord256Label3-mirror-auto1-ada-target0.7-bgc-resumeffhq256'
model=$path/network-snapshot-001011.pkl 

python encode_images.py $imgdir/aligned_images/ $imgdir/generated_images/ $imgdir/latent_representations_w/ \
--vgg16-pkl /data/duongdb/vgg16_zhang_perceptual.pkl \
--network_pkl $model

# python project_images.py $imgdir/aligned_images/ $imgdir/generated_images/ \
# --vgg16-pkl /data/duongdb/vgg16_zhang_perceptual.pkl --network-pkl $model
