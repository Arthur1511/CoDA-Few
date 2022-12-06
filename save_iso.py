"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, get_val_data_loaders, prepare_sub_folder, get_config, write_2images, write_images, write_loss, Timer
import argparse
import numpy as np
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer, FUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import shutil
from glob import glob


cudnn.benchmark = True
path = "config/lungs/*.yaml"

models= {}
for f in glob(path):
    
    # Load experiment setting
    config = get_config(f)

    cuda0 = torch.device('cuda:2')
    cuda1 = torch.device('cuda:2')
    devices = (cuda0, cuda1)


    model_name = os.path.splitext(os.path.basename(f))[0]
    output_directory = os.path.join('lungs3' + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    tissue = model_name.split("_")[1]
    target = model_name.split("_")[-1].upper()
    source = model_name.split("_")[-2].upper()

    # Setup model and data loader
    trainer = FUNIT_Trainer(config, devices, tissue)
    # trainer.cuda(cuda0)

    val_loader_list = get_val_data_loaders(config)

    # Start training
    iterations = trainer.resume(
        checkpoint_directory, hyperparameters=config)


    lab_list = list()
    dataset_jacc_dict = dict()
    
    print("MODEL:", model_name)
    print()
    
    for l in sorted(source+target):

        i = config['datasets_test'].index(l)

        print('\n    Generating Isomorphic Represetantions ' + config['datasets_test'][i] + '...')

        dataset_jacc = list()

        for it, data in enumerate(val_loader_list[i]):

            x = data[0].cuda(cuda0).detach()
            y = data[1].cuda(cuda1).detach()
            path = data[2]

            iso = trainer.get_iso(x, y, i, config)
            dataset_jacc.append(iso)

        dataset_jacc = np.asarray(dataset_jacc)

        dataset_jacc_dict[l] = dataset_jacc

        print('        Finished ' + config['datasets_test'][i])

    models[model_name] = dataset_jacc_dict

np.savez("funit_iso.npz", **models)