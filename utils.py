"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.utils as vutils
import yaml
from skimage.measure import label, regionprops
from torch.autograd import Variable
from torch.optim import lr_scheduler
# from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from torchvision import transforms

from data import ImageFilelist, ImageFolder, ValImageFolder
from networks import Vgg16

# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# load_inception
# vgg_preprocess
# get_scheduler
# weights_init


def get_all_data_loaders(conf):

    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    resize_to = (conf['resize_height'], conf['resize_width'])
    crop_to = (conf['crop_height'], conf['crop_width'])

    train_loader = get_data_loader_folder(conf['data_root'], 'train', len(conf['datasets_train']), conf['datasets_train'],
                                          conf['label_use_train'], batch_size, False, True, resize_to, crop_to, num_workers)
    test_loader = get_data_loader_folder(conf['data_root'], 'test', len(conf['datasets_test']), conf['datasets_test'],
                                         conf['label_use_test'], batch_size, True, False, crop_to, crop_to, num_workers)

    return train_loader, test_loader


def get_data_loader_folder(data_root, mode, n_datasets, letters, label_use, batch_size, return_paths, shuffle, resize_to=(284, 284), crop_to=(256, 256), num_workers=4):

    dataset = ImageFolder(data_root, mode, n_datasets, letters,
                          label_use, resize_to, crop_to, return_paths)
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=shuffle, drop_last=True, num_workers=num_workers)

    return loader


def get_val_data_loaders(conf, mode='test'):
    # if test_train:
    #     batch_size = 1
    # else:
    #     batch_size = conf['batch_size']
    batch_size = 1
    num_workers = conf['num_workers']
    dataset_letters = conf['datasets_test']
    resize_to = (conf['resize_height'], conf['resize_width'])
    crop_to = (conf['crop_height'], conf['crop_width'])

    val_loader_list = list()

    for i in range(len(dataset_letters)):

        # val_loader = get_data_loader_folder(os.path.join(conf['data_root']), 'test', dataset_letters[i], 1,
        #                                      True, trim, num_workers, sample=1.0, return_path=True, random_transform=0, channels=conf['input_dim'])
        val_dataset = ValImageFolder(
            conf['data_val'], mode, dataset_letters[i], resize_to=resize_to, crop_to=crop_to, return_paths=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=True, num_workers=num_workers)
        val_loader_list.append(val_loader)

    return val_loader_list


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def eformat(f, prec):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d" % (mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):

    # expand gray-scale images to 3 channels
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    image_tensor = torch.cat([images[:display_image_num]
                              for images in image_outputs], 0)
    image_grid = vutils.make_grid(
        image_tensor.data, nrow=display_image_num, padding=0, normalize=True)

    vutils.save_image(image_grid, file_name, nrow=1)


def process_label(x, tissue="lungs"):
    x[x > 0] = 1
    msk = np.zeros_like(x)
    labels = label(x)
    regions = regionprops(labels)
    area = sorted([p.area for p in regions], reverse=True)

    if len(area) < 2:
        return x

    else:
        region_pos = 1 if tissue == "lungs" else 0

        for R in regions:
            if R.area >= area[region_pos]:
                # draw the region (I'm sure there's a more efficient way of doing it)
                for c in R.coords:
                    msk[c[0], c[1]] = 1
        return msk


def write_2images(image_outputs, display_image_num, image_directory, ind_a, ind_b, postfix):

    n = len(image_outputs)

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_%s2%s_%s.png' %
                   (image_directory, letters[ind_a], letters[ind_b], postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_%s2%s_%s.png' %
                   (image_directory, letters[ind_b], letters[ind_a], postfix))


def write_images(image_outputs, display_image_num, letters, image_directory, ind, postfix):

    #     letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    __write_images(image_outputs, display_image_num,
                   '%s/gen_%s_%s.png' % (image_directory, letters[ind], postfix))


def write_1images(image_outputs, display_image_num, image_directory, ind_a, ind_b, postfix):

    n = len(image_outputs)

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    __write_images(image_outputs, display_image_num, '%s/%s_gen_%s2%s.png' %
                   (image_directory, postfix, letters[ind_a], letters[ind_b]))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" %
                    (iterations, img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations,
                       '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations,
                       '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(
                html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(
                html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(
                html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(
                html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low),
                             high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


# def load_vgg16(model_dir):
#     """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
#         if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
#             os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
#         vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
#         vgg = Vgg16()
#         for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
#             dst.data[:] = src
#         torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
#     vgg = Vgg16()
#     vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
#     return vgg

def load_inception(model_path):
    state_dict = torch.load(model_path)
    model = inception_v3(pretrained=False, transform_input=True)
    model.aux_logits = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, state_dict['fc.weight'].size(0))
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    return model


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def pytorch03_to_pytorch04(state_dict_base, trainer_name):
    def __conversion_core(state_dict_base, trainer_name):
        state_dict = state_dict_base.copy()
        if trainer_name == 'MUNIT':
            for key, value in state_dict_base.items():
                if key.endswith(('enc_content.model.0.norm.running_mean',
                                 'enc_content.model.0.norm.running_var',
                                 'enc_content.model.1.norm.running_mean',
                                 'enc_content.model.1.norm.running_var',
                                 'enc_content.model.2.norm.running_mean',
                                 'enc_content.model.2.norm.running_var',
                                 'enc_content.model.3.model.0.model.1.norm.running_mean',
                                 'enc_content.model.3.model.0.model.1.norm.running_var',
                                 'enc_content.model.3.model.0.model.0.norm.running_mean',
                                 'enc_content.model.3.model.0.model.0.norm.running_var',
                                 'enc_content.model.3.model.1.model.1.norm.running_mean',
                                 'enc_content.model.3.model.1.model.1.norm.running_var',
                                 'enc_content.model.3.model.1.model.0.norm.running_mean',
                                 'enc_content.model.3.model.1.model.0.norm.running_var',
                                 'enc_content.model.3.model.2.model.1.norm.running_mean',
                                 'enc_content.model.3.model.2.model.1.norm.running_var',
                                 'enc_content.model.3.model.2.model.0.norm.running_mean',
                                 'enc_content.model.3.model.2.model.0.norm.running_var',
                                 'enc_content.model.3.model.3.model.1.norm.running_mean',
                                 'enc_content.model.3.model.3.model.1.norm.running_var',
                                 'enc_content.model.3.model.3.model.0.norm.running_mean',
                                 'enc_content.model.3.model.3.model.0.norm.running_var',
                                 )):
                    del state_dict[key]
        else:
            def __conversion_core(state_dict_base):
                state_dict = state_dict_base.copy()
                for key, value in state_dict_base.items():
                    if key.endswith(('enc.model.0.norm.running_mean',
                                     'enc.model.0.norm.running_var',
                                     'enc.model.1.norm.running_mean',
                                     'enc.model.1.norm.running_var',
                                     'enc.model.2.norm.running_mean',
                                     'enc.model.2.norm.running_var',
                                     'enc.model.3.model.0.model.1.norm.running_mean',
                                     'enc.model.3.model.0.model.1.norm.running_var',
                                     'enc.model.3.model.0.model.0.norm.running_mean',
                                     'enc.model.3.model.0.model.0.norm.running_var',
                                     'enc.model.3.model.1.model.1.norm.running_mean',
                                     'enc.model.3.model.1.model.1.norm.running_var',
                                     'enc.model.3.model.1.model.0.norm.running_mean',
                                     'enc.model.3.model.1.model.0.norm.running_var',
                                     'enc.model.3.model.2.model.1.norm.running_mean',
                                     'enc.model.3.model.2.model.1.norm.running_var',
                                     'enc.model.3.model.2.model.0.norm.running_mean',
                                     'enc.model.3.model.2.model.0.norm.running_var',
                                     'enc.model.3.model.3.model.1.norm.running_mean',
                                     'enc.model.3.model.3.model.1.norm.running_var',
                                     'enc.model.3.model.3.model.0.norm.running_mean',
                                     'enc.model.3.model.3.model.0.norm.running_var',

                                     'dec.model.0.model.0.model.1.norm.running_mean',
                                     'dec.model.0.model.0.model.1.norm.running_var',
                                     'dec.model.0.model.0.model.0.norm.running_mean',
                                     'dec.model.0.model.0.model.0.norm.running_var',
                                     'dec.model.0.model.1.model.1.norm.running_mean',
                                     'dec.model.0.model.1.model.1.norm.running_var',
                                     'dec.model.0.model.1.model.0.norm.running_mean',
                                     'dec.model.0.model.1.model.0.norm.running_var',
                                     'dec.model.0.model.2.model.1.norm.running_mean',
                                     'dec.model.0.model.2.model.1.norm.running_var',
                                     'dec.model.0.model.2.model.0.norm.running_mean',
                                     'dec.model.0.model.2.model.0.norm.running_var',
                                     'dec.model.0.model.3.model.1.norm.running_mean',
                                     'dec.model.0.model.3.model.1.norm.running_var',
                                     'dec.model.0.model.3.model.0.norm.running_mean',
                                     'dec.model.0.model.3.model.0.norm.running_var',
                                     )):
                        del state_dict[key]
        return state_dict

    state_dict = dict()
    state_dict['a'] = __conversion_core(state_dict_base['a'], trainer_name)
    state_dict['b'] = __conversion_core(state_dict_base['b'], trainer_name)
    return state_dict

# Implementation from: https://github.com/hubutui/DiceLoss-PyTorch

# Implementation from: https://github.com/hubutui/DiceLoss-PyTorch


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) +
                        target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(
                            target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def dice_loss(prd, lab, device, weight, batch_size, num_classes=2, ignore_index=-1):

    criterion = DiceLoss(weight=weight, ignore_index=ignore_index).cuda(device)
    one_hot_labs = F.one_hot(
        lab, num_classes=num_classes)

    one_hot_labs = one_hot_labs.reshape((batch_size, 2, 256, 256))

    return criterion(prd, one_hot_labs)
