# Download the model of choice
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
from io import BytesIO
import IPython.display
from math import ceil
from PIL import Image, ImageDraw
import os
import pickle
from utils import log_progress, imshow

class Mosaic:
    def __init__(self, path):
        dnnlib.tflib.init_tf()

        print('Loading networks from "%s"...' % path)
        with dnnlib.util.open_url(path) as fp:
            self._G, self._D, self.Gs = pickle.load(fp)
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
    # Generates a list of images, based on a list of latent vectors (Z), and a list (or a single constant) of truncation_psi's.
    def generate_images_in_w_space(self, dlatents, truncation_psi):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        Gs_kwargs.truncation_psi = truncation_psi
        # dlatent_avg = self.Gs.get_var('dlatent_avg') # [component]

        imgs = []
        for _, dlatent in log_progress(enumerate(dlatents), name = "Generating images"):
            #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
            # dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
            row_images = self.Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
            imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
        return imgs       

    def generate_images(self, zs, truncation_psi):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if not isinstance(truncation_psi, list):
            truncation_psi = [truncation_psi] * len(zs)
            
        imgs = []
        for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
            Gs_kwargs.truncation_psi = truncation_psi[z_idx]
            noise_rnd = np.random.RandomState(1) # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
            images = self.Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
        return imgs

    def generate_zs_from_seeds(self, seeds):
        zs = []
        for _, seed in enumerate(seeds):
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *self.Gs.input_shape[1:]) # [minibatch, component]
            zs.append(z)
        return zs

    # Generates a list of images, based on a list of seed for latent vectors (Z), and a list (or a single constant) of truncation_psi's.
    def generate_images_from_seeds(self, seeds, truncation_psi):
        return imshow(self.generate_images(self.generate_zs_from_seeds(seeds), truncation_psi))


    def convertZtoW(self, latent, truncation_psi=0.7, truncation_cutoff=9):
        dlatent = self.Gs.components.mapping.run(latent, None) # [seed, layer, component]
        dlatent_avg = self.Gs.get_var('dlatent_avg') # [component]
        for i in range(truncation_cutoff):
            dlatent[0][i] = (dlatent[0][i]-dlatent_avg)*truncation_psi + dlatent_avg
            
        return dlatent

    def interpolate(self, zs, steps):
        out = []
        for i in range(len(zs)-1):
            for index in range(steps):
                fraction = index/float(steps) 
                out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
        return out