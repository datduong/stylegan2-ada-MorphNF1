

import sys,os,re,pickle
from PIL import Image
import numpy as np 


# path='/data/duongdb/NF1BeforeAfterInter02012021/Crop/training-stylegan2-ada-Lab3-BaseLabEmb/00000-Tfrecord256wLabel-mirror-auto1-ada-target0.7-bgc-resumeffhq256'

path = sys.argv[1]

mix_ratio = sys.argv[2].strip().split() 

outdir = os.path.join(path,'InterpolateFakeImg')
if not os.path.exists(outdir): 
  os.makedirs(outdir)

for seed in np.arange(0,200): 
  image_list = [os.path.join(path,'SameZvecMix'+str(m),f'seed{seed:04d}.png') for m in mix_ratio ]
  images = [Image.open(x) for x in image_list]
  widths, heights = zip(*(i.size for i in images))
  total_width = sum(widths)
  max_height = max(heights)
  new_im = Image.new('RGB', (total_width, max_height))
  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  #
  new_im.save(os.path.join(outdir,f'seed{seed:04d}.png'))


