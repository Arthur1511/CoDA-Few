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
from tensorboardX import SummaryWriter
import shutil

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
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config, devices)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config, devices)
elif opts.trainer == 'FUNIT':
    trainer = FUNIT_Trainer(config, devices, tissue)
else:
    sys.exit("Only support MUNIT|UNIT|FUNIT")
# trainer.cuda(cuda0)

train_loader, test_loader = get_all_data_loaders(config)
val_loader_list = get_val_data_loaders(config)

# train_img_samples = []
# train_msk_samples = []
test_img_samples = []
test_msk_samples = []

for d in range(len(config['datasets_test'])):

#     train_samples = train_loader.dataset.load_samples(display_size, d)
    test_samples = test_loader.dataset.load_samples(display_size, d)

#     train_img_samples.append(torch.stack(train_samples[0]))
#     train_msk_samples.append(torch.stack(train_samples[1]))
    test_img_samples.append(torch.stack(test_samples[0]))
    test_msk_samples.append(torch.stack(test_samples[1]))

# train_display_images_list = [torch.stack(train_loader.dataset.load_samples(display_size, d)[0]) for d in range(config['n_datasets'])]
# test_display_images_list = [torch.stack(test_loader.dataset.load_samples(display_size, d)[0]) for d in range(config['n_datasets'])]


# Creating logs directory
if not os.path.exists(log_directory):
    print("Creating directory: {}".format(log_directory))
    os.makedirs(log_directory)

train_writer = SummaryWriter(os.path.join(
    log_directory, model_name), flush_secs=10)
# copy config file to output folder
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Setup logger and output folders
# model_name = os.path.splitext(os.path.basename(opts.config))[0]
# output_directory = os.path.join(opts.output_path + "/outputs", model_name)
# checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# log_directory = os.path.join(output_directory, 'logs')

# Start training
iterations = trainer.resume(
    checkpoint_directory, hyperparameters=config) if opts.resume else 0

# letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
letters = config['datasets_train']

print('Training starts...')
while True:

    for it, data in enumerate(train_loader):

        img_a = data[0].cuda(cuda0).detach()
        img_b = data[1].cuda(cuda0).detach()

        msk_a = data[2].cuda(cuda1).detach()
        msk_b = data[3].cuda(cuda1).detach()

        if opts.trainer == 'FUNIT':
            ind_a = data[4]
            ind_b = data[5]
        else:
            ind_a = data[4][0].item()
            ind_b = data[5][0].item()

        lab_a = data[6][0]
        lab_b = data[7][0]

        # with Timer("Elapsed time in update: %f"):
        # Translation forward and backward.
        trainer.dis_update(img_a, img_b, ind_a, ind_b, config)
        gen_losses = trainer.gen_update(img_a, img_b, ind_a, ind_b, config)
        torch.cuda.synchronize()

        # Supervised forward and backward.
        sup_losses = trainer.sup_update(
            img_a, img_b, msk_a, msk_b, ind_a, ind_b, config)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:

            # recon_x_w_a, recon_s_w_a, recon_c_w_a, recon_x_w_b, recon_s_w_b, recon_c_w_b, recon_x_cyc_w_a, recon_x_cyc_w_b, loss_gen = gen_losses
            # sup_a, sup_b, sup_a_recon, sup_b_recon, sup_loss = sup_losses

            write_loss(iterations, trainer, train_writer)

            if opts.trainer == 'FUNIT':
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

                    # print('    Sup Losses: a %.3f, a_recon %.3f, loss_sup %.3f' % (
                    #     sup_losses))
                    # print('    Sup Losses: a %.3f, a_recon %.3f, loss_sup %.3f' % (
                    #     sup_losses), file=open(opts.output_path + '/logs/' + opts.config.split('/')[-1].replace('.yaml', '_loss.log'), 'a'))
            else:
                print('Iteration: %08d/%08d, datasets [%s, %s]' % (
                    iterations + 1, max_iter, letters[ind_a], letters[ind_b]))
                print('    Gen Losses: x_w_a %.3f, s_w_a %.3f, c_w_a %.3f, x_w_b %.3f, s_w_b %.3f, c_w_b %.3f, x_cyc_w_a %.3f, x_cyc_w_b %.3f, loss_gen %.3f' % (gen_losses))
                if lab_a or lab_b:
                    print('    Sup Losses: a %.3f, b %.3f, a_recon %.3f, b_recon %.3f, loss_sup %.3f' % (
                        sup_losses))
            sys.stdout.flush()

        if (iterations + 1) % config['image_display_iter'] == 0:
            if opts.trainer == 'FUNIT':
                with torch.no_grad():
                    #     img_outputs = trainer.sample(test_img_samples[ind_a[0]].cuda(cuda0),
                    #                                  test_img_samples[ind_b[0]].cuda(
                    #                                      cuda0),
                    #                                  test_msk_samples[ind_a[0]].cuda(
                    #                                      cuda1),
                    #                                  test_msk_samples[ind_b[0]].cuda(
                    #                                      cuda1),
                    #                                  ind_a,
                    #                                  ind_b)
                    # write_1images(img_outputs, display_size, image_directory,
                    #               ind_a[0], ind_b[0], 'img_it_%07d' % (iterations + 1))
                    for d_index in range(len(test_img_samples)):
                        img_outputs = trainer.sample(test_img_samples[d_index].cuda(cuda0),
                                                     test_msk_samples[d_index].cuda(
                            cuda1),
                            d_index)
                        write_images(img_outputs, display_size, config['datasets_test'], image_directory,
                                     d_index, 'img_it_%07d' % (iterations + 1))
            else:
                with torch.no_grad():
                    img_outputs = trainer.sample(test_img_samples[ind_a].cuda(cuda0),
                                                 test_img_samples[ind_b].cuda(
                        cuda0),
                        test_msk_samples[ind_a].cuda(
                        cuda1),
                        test_msk_samples[ind_b].cuda(
                        cuda1),
                        ind_a,
                        ind_b)
                write_2images(img_outputs, display_size, image_directory,
                              ind_a, ind_b, 'img_it_%07d' % (iterations + 1))

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

        if opts.trainer != 'FUNIT':
            trainer.update_learning_rate()
