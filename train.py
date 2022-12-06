"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from trainer import FUNIT_Trainer
from utils import (Timer, get_all_data_loaders, get_config,
                   get_val_data_loaders, prepare_sub_folder, write_2images,
                   write_images, write_loss)

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import shutil
import sys

from tensorboardX import SummaryWriter

# python train.py --config config/funit_cxr_lungs.yaml --trainer FUNIT --output_path teste

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT',
                    help="MUNIT|UNIT|FUNIT")
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

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
tissue = model_name.split("_")[1]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
log_directory = os.path.join(output_directory, 'logs')

# Setup model and data loader

trainer = FUNIT_Trainer(config, devices, tissue)

train_loader, test_loader = get_all_data_loaders(config)
val_loader_list = get_val_data_loaders(config)


test_img_samples = []
test_msk_samples = []

for d in range(len(config['datasets_test'])):

    test_samples = test_loader.dataset.load_samples(display_size, d)

    test_img_samples.append(torch.stack(test_samples[0]))
    test_msk_samples.append(torch.stack(test_samples[1]))


# Creating logs directory
if not os.path.exists(log_directory):
    print("Creating directory: {}".format(log_directory))
    os.makedirs(log_directory)

train_writer = SummaryWriter(os.path.join(
    log_directory, model_name), flush_secs=10)
# copy config file to output folder
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Setup logger and output folders


# Start training
iterations = trainer.resume(
    checkpoint_directory, hyperparameters=config) if opts.resume else 0

letters = config['datasets_train']

print('Training starts...')
while True:

    for it, data in enumerate(train_loader):

        img_a = data[0].cuda(cuda0).detach()
        img_b = data[1].cuda(cuda0).detach()

        msk_a = data[2].cuda(cuda1).detach()
        msk_b = data[3].cuda(cuda1).detach()

        ind_a = data[4]
        ind_b = data[5]

        lab_a = data[6][0]
        lab_b = data[7][0]

        # Translation forward and backward.
        trainer.dis_update(img_a, img_b, ind_a, ind_b, config)
        gen_losses = trainer.gen_update(img_a, img_b, ind_a, ind_b, config)
        torch.cuda.synchronize()

        # Supervised forward and backward.
        sup_losses = trainer.sup_update(
            img_a, img_b, msk_a, msk_b, ind_a, ind_b, config)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:

            write_loss(iterations, trainer, train_writer)

            print('\nIteration: %08d/%08d, datasets [%s, %s]' % (
                iterations + 1, max_iter, letters[ind_a[0].item()], letters[ind_b[0].item()]))
            print('\nIteration: %08d/%08d, datasets [%s, %s]' % (
                iterations + 1, max_iter, letters[ind_a[0].item()], letters[ind_b[0].item()]), file=open(os.path.join(log_directory, opts.config.split('/')[-1].replace('.yaml', '_loss.log')), 'a'))

            print('    Gen Losses: recon_x %.3f, recon_c %.3f, recon_s %.3f, gen_adv %.3f, acc_gen_adv %.3f, gen_total %.3f' % (
                gen_losses))
            print('    Gen Losses: recon_x %.3f, recon_c %.3f, recon_s %.3f, gen_adv %.3f, acc_gen_adv %.3f, gen_total %.3f' % (
                gen_losses), file=open(os.path.join(log_directory, opts.config.split('/')[-1].replace('.yaml', '_loss.log')), 'a'))
            if lab_a or lab_b:
                print('    Sup Losses: a %.3f, b %.3f, a_recon %.3f, b_recon %.3f, loss_sup %.3f' % (
                    sup_losses))
                print('    Sup Losses: a %.3f, b %.3f, a_recon %.3f, b_recon %.3f, loss_sup %.3f' % (
                    sup_losses), file=open(os.path.join(log_directory, opts.config.split('/')[-1].replace('.yaml', '_loss.log')), 'a'))

            sys.stdout.flush()

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():

                for d_index in range(len(test_img_samples)):
                    img_outputs = trainer.sample(test_img_samples[d_index].cuda(cuda0),
                                                 test_msk_samples[d_index].cuda(cuda1), d_index)
                    write_images(img_outputs, display_size, config['datasets_test'], image_directory,
                                 d_index, 'img_it_%07d' % (iterations + 1))

        if (iterations + 1) % config['val_iter'] == 0:
            jacc_file = open(os.path.join(log_directory, opts.config.split(
                '/')[-1].replace('.yaml', '_jacc.log')), 'a')

            lab_list = list()

            for i in range(len(config['datasets_test'])):

                print('\n    Testing ' + config['datasets_test'][i] + '...')

                dataset_jacc = list()

                for it, data in enumerate(val_loader_list[i]):

                    x = data[0].cuda(cuda0).detach()
                    y = data[1].cuda(cuda1).detach()
                    path = data[2]

                    jacc, p, = trainer.sup_forward(x, y, i, config)
                    dataset_jacc.append(jacc)

                dataset_jacc = np.asarray(dataset_jacc)

                print('        Test ' + config['datasets_test'][i] + ' Jaccard iteration ' + str(iterations + 1) + ': ' + str(
                    100 * dataset_jacc.mean()) + ' +/- ' + str(100 * dataset_jacc.std()))

                jacc_file.write('        Test ' + config['datasets_test'][i] + ' Jaccard iteration ' + str(
                    iterations + 1) + ': ' + str(100 * dataset_jacc.mean()) + ' +/- ' + str(100 * dataset_jacc.std()) + '\n')

                train_writer.add_scalar("Jaccard Dataset {}".format(
                    config['datasets_test'][i]), (100 * dataset_jacc.mean()), iterations + 1)
            jacc_file.close()
        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            print('Finish training')
            train_writer.flush()
            train_writer.close()
            sys.exit(0)
