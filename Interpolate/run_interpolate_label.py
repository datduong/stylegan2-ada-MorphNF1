
# ! run these lines, we can just run interactive
"""
sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada/Interpolate 
"""


import os
import re
import sys
import pickle
import argparse
import numpy as np
import PIL.Image as Image
from tqdm import tqdm 

sys.path.append('/data/duongdb/stylegan2-ada/')

import dnnlib
import dnnlib.tflib as tflib

from dnnlib import EasyDict
from training import dataset
from training import networks

import pretrained_networks
import tensorflow as tf


# ! simple interpolation without face-action direction vector, based on run_generator.py

import interpolate_util as interpolate


main_path = sys.argv[1] # '/data/duongdb/NF1BeforeAfterInter02012021/Crop/training-stylegan2-ada-Lab3-NoLabEmb/00000-Tfrecord256NoLabel-mirror-auto1-ada-target0.7-bgc-resumeffhq256'

os.chdir(main_path)
out_path = os.path.join(main_path,'Interpolate')
if not os.path.exists(out_path): 
    os.makedirs ( out_path )

# ! change path ? 
os.chdir(out_path)
model_path = os.path.join(main_path, sys.argv[2]) # ! put in model pickle to load

fps = 20
results_size = 256

# Gs, noise_vars, Gs_kwargs = interpolate.load_model(model_path,truncation_psi=0.7) # load model
# print ('Gs_kwargs')
# print (Gs_kwargs)

# ----------------------------------------------------------------------------
# ! the same random z, interpolate label vec

seeds = np.arange(0,50)
for seed_idx, seed in enumerate(seeds):
    interpolate.make_labelvec_interp_png(seed, 10, 'seed'+str(seed), model_path, 256, truncation_psi=0.7)


# ----------------------------------------------------------------------------


# # ! latent vectors inferred from real images

# latent_codes1, latent_files1 = interpolate.get_final_latents(sys.argv[3])  # Before images '/data/duongdb/stylegan2-ada/results/00010-project-real-images'
# len(latent_codes1), latent_codes1[0].shape, latent_codes1[1].shape

# latent_codes2, latent_files2 = interpolate.get_final_latents(sys.argv[4])  # After images '/data/duongdb/stylegan2-ada/results/00009-project-real-images'
# len(latent_codes2), latent_codes2[0].shape, latent_codes2[1].shape

# # ! latent_codes[0] is the same random vector, get duplicated serveral times, one for each layer in stylegan2

# counter = 1
# for index1 in range (len(latent_files1)): 
#     images = interpolate.generate_image_from_projected_latents(latent_codes1[index1], Gs, Gs_kwargs)
#     recreated_img1 = Image.fromarray(images[0]).resize((results_size, results_size))
#     for index2 in range (len(latent_files2)): 
#         images = interpolate.generate_image_from_projected_latents(latent_codes2[index2], Gs, Gs_kwargs)
#         recreated_img2 = Image.fromarray(images[0]).resize((results_size, results_size))
#         interpolate.make_latent_interp_png_real_img(latent_codes1[index1], latent_codes2[index2], recreated_img1, recreated_img2, 10, 'ExampleReconInterAfterPair'+str(counter), Gs, Gs_kwargs, results_size)
#         interpolate.make_latent_interp_animation_real_img(latent_codes1[index1], latent_codes2[index2], recreated_img1, recreated_img2, 200, 'ExampleReconInterAfterPair'+str(counter), fps, Gs, Gs_kwargs, results_size)
#         counter = counter + 1


