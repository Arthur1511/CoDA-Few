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
path = "config/" + sys.argv[1] + "/*.yaml"
# task_folder = 'lungs3'
task_folder = sys.argv[2]

models= {}
for f in glob(path):
    
    # Load experiment setting
    config = get_config(f)

    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:0')
    devices = (cuda0, cuda1)


    model_name = os.path.splitext(os.path.basename(f))[0]
    output_directory = os.path.join(task_folder + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    tissue = model_name.split("_")[1]
    target = model_name.split("_")[-1].upper()
    source = model_name.split("_")[-2].upper()
    
    sample_directory = os.path.join('samples', task_folder, model_name)
    
    if not os.path.exists(sample_directory):
        print("Creating directory: {}".format(sample_directory))
        os.makedirs(sample_directory)

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

        print('\n    Generating Samples ' + config['datasets_test'][i] + '...')

        dataset_jacc = list()

        for it, data in enumerate(val_loader_list[i]):

            x = data[0].cuda(cuda0).detach()
            y = data[1].cuda(cuda1).detach()
            path = data[2]

            _, pred = trainer.sup_forward(x, y, i, config)
            
            x = x.cpu()
            y = y.cpu().reshape((1, 1, 256, 256))
            pred = torch.from_numpy(pred.reshape((1, 1, 256, 256)))
            
            img_outputs = (x, y, pred)
            
#             print(x.cpu().shape, y.cpu().reshape((1, 1, 256, 256)).shape, pred.reshape((1, 1, 256, 256)).shape)
            
            write_images(img_outputs, 3, config['datasets_test'], sample_directory,
                                     i, path[0].split(".")[0])
            
        print('        Finished ' + config['datasets_test'][i])
