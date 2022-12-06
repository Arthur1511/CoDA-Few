"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from trainer import MUNIT_Trainer, UNIT_Trainer, FUNIT_Trainer
from utils import get_config, get_val_data_loaders

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import argparse
import os
import sys

# import tensorboardX
from skimage import io

# python test.py --config config/funit_cxr_lungs.yaml --trainer FUNIT --output_path teste_abc_efg

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--generate_pred", action="store_true")
parser.add_argument(
    "--mode", help="Set to generate predictions (train or test)", default="test", type=str)
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:0')
devices = (cuda0, cuda1)

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config, devices)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config, devices)
elif opts.trainer == 'FUNIT':
    trainer = FUNIT_Trainer(config, devices)
else:
    sys.exit("Only support MUNIT|UNIT|FUNIT")
# trainer.cuda(cuda0)
config['data_root'] = "../dataset_hopping/Datasets/CXR_lungs"
# train_loader, test_loader = get_all_data_loaders(config)

val_loader_list = get_val_data_loaders(config, mode=opts.mode)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory = os.path.join(output_directory, 'checkpoints')

# Start testing
iterations = trainer.resume(checkpoint_directory, hyperparameters=config)

letters = config['datasets_test']

jacc_file = open(opts.output_path + '/logs/' + opts.config.split('/')
                 [-1].replace('.yaml', '_jacc_test.log'), 'a')


lab_list = list()

for i in range(len(letters)):

    print('\n    Testing ' + letters[i] + '...')

    dataset_jacc = list()

    for it, data in enumerate(val_loader_list[i]):

        x = data[0].cuda(cuda0).detach()
        y = data[1].cuda(cuda1).detach()
        path = data[2]

        jacc, p, = trainer.sup_forward(x, y, i, config)
        dataset_jacc.append(jacc)

        if opts.generate_pred:
            image_directory = 'hopping'
            if not os.path.exists(os.path.join(image_directory, 'label_' + opts.mode + letters[i])):
                os.mkdir(os.path.join(image_directory,
                                      'label_' + opts.mode + letters[i]))
            if not os.path.exists(os.path.join(image_directory, opts.mode + letters[i])):
                os.mkdir(os.path.join(image_directory, opts.mode + letters[i]))

            x_path = os.path.join(
                image_directory, opts.mode + letters[i], path)
            p_path = os.path.join(
                image_directory, 'label_' + opts.mode + letters[i], path)

            np_x = x.cpu().numpy().squeeze()
            np_y = y.cpu().numpy().squeeze()

            if not os.path.isfile(x_path):
                io.imsave(x_path, np_x)
            if not os.path.isfile(p_path):
                io.imsave(p_path, p)
        



    dataset_jacc = np.asarray(dataset_jacc)


    print('        Test ' + letters[i] + ' Jaccard iteration ' + str(iterations) + ': ' + str(
        100 * dataset_jacc.mean()) + ' +/- ' + str(100 * dataset_jacc.std()))

    jacc_file.write('        Test ' + letters[i] + ' Jaccard iteration ' + str(
        iterations) + ': ' + str(100 * dataset_jacc.mean()) + ' +/- ' + str(100 * dataset_jacc.std()) + '\n')


jacc_file.close()
