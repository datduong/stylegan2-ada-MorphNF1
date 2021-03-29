import os
import re
import sys
import pickle

import argparse
import numpy as np

import PIL.Image as Image

sys.path.append('/data/duongdb/stylegan2-ada')

import dnnlib
import dnnlib.tflib as tflib

from pathlib import Path

from dnnlib import EasyDict
from training import dataset
from training import networks

import pretrained_networks
import tensorflow as tf

from tqdm import tqdm 

main_data_dir = '/data/duongdb/WS_Aging02112021/generated_images' 
# network_pkl = '/data/duongdb/stylegan2-ffhq-config-f.pkl'
network_pkl = '/data/duongdb/WS_Aging02112021/stylegan2-ada-Lab3-aveLabEmb/00000-ImgAlignTfrecord256Label3-mirror-auto1-ada-target0.7-bgc-resumeffhq256/network-snapshot-001264.pkl'


truncation_psi = 0.7 # ! default suggested for face 
output_gifs_path = main_data_dir

fps = 15
results_size = 256

# ----------------------------------------------------------------------------
# ! load model 

tflib.init_tf()
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as fp:
    _G, _D, Gs = pickle.load(fp)

Gs_kwargs = { # ! follow generate.py
    'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
    'randomize_noise': False
}
if truncation_psi is not None:
    Gs_kwargs['truncation_psi'] = truncation_psi

noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]


# ----------------------------------------------------------------------------

def generate_image_from_projected_latents(latent_vector):
    images = Gs.components.synthesis.run(latent_vector, **Gs_kwargs)
    return images

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_control_latent_vectors(path):
    files = [x for x in Path(path).iterdir() if str(x).endswith('.npy')]
    latent_vectors = {f.name[:-4]:np.load(f) for f in files}
    return latent_vectors

def make_latent_control_animation(feature, start_amount, end_amount, step_size, person):  
    all_imgs = [] 
    all_imgs2 = None # just record the list
    amounts = np.linspace( start_amount, end_amount, int (abs(end_amount-start_amount)/step_size) )
    save_every = int ( len(amounts)//10 )
    for counter, amount_to_move in tqdm(enumerate(amounts)):
        modified_latent_code = np.array(latent_code_to_use)
        modified_latent_code += latent_controls[feature]*amount_to_move
        images = generate_image_from_projected_latents(modified_latent_code)
        latent_img = Image.fromarray(images[0]).resize((results_size, results_size))
        all_imgs.append(get_concat_h(image_to_use, latent_img))
        if counter == 0: 
            all_imgs2 = latent_img
        elif counter % save_every == 0: 
            all_imgs2 = get_concat_h(all_imgs2, latent_img)
        else: 
            pass
    #
    save_name = output_gifs_path+'/{0}_{1}.gif'.format(person, feature)
    # print ('now save')
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)
    all_imgs2.save(output_gifs_path+'/{0}_{1}.png'.format(person, feature))


# ----------------------------------------------------------------------------

latent_controls = get_control_latent_vectors('/data/duongdb/stylegan2directions/')
# len(latent_controls), latent_controls.keys(), latent_controls['age'].shape

for f in os.listdir(main_data_dir): 

    if f.endswith('.npy'): 

        fname = re.sub('.npy','',f) # get just the name
        latent_code_to_use = np.load(os.path.join(main_data_dir,f)) 
        latent_code_to_use = latent_code_to_use.reshape ((1,18,512))

        image_to_use = generate_image_from_projected_latents(latent_code_to_use)
        image_to_use = Image.fromarray(image_to_use[0]).resize((results_size, results_size))

        make_latent_control_animation(feature='age', start_amount=0, end_amount=20, step_size=0.2, person=fname+'Plus')

        make_latent_control_animation(feature='age', start_amount=-20, end_amount=0, step_size=0.2, person=fname+'Minus')

        make_latent_control_animation(feature='gender', start_amount=0, end_amount=20, step_size=0.2, person=fname+'Plus')
        
        make_latent_control_animation(feature='gender', start_amount=-20, end_amount=0, step_size=0.2, person=fname+'Minus')

