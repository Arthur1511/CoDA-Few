"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import time
import os.path
import os
import torch
import numpy as np

from skimage import io
from skimage import transform, util

def default_loader(path):

    img = io.imread(path)

    return img


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(
            list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]                             : i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]])
                     for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(img_dir, msk_dir):

    samples = []

    for f in sorted(os.listdir(img_dir)):

        img_path = os.path.join(img_dir, f)
        msk_path = os.path.join(msk_dir, f)

        if os.path.isfile(img_path):
            samples.append((img_path, msk_path))

    return samples


class ImageFolder(data.Dataset):

    def __init__(self, data_root, mode, n_datasets, letters, label_use, resize_to=(284, 284), crop_to=(256, 256), return_paths=False):

        #         letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        roots = [(os.path.join(data_root, '%s%s' % (mode, letters[d])), os.path.join(
            data_root, 'label_%s%s' % (mode, letters[d]))) for d in range(n_datasets)]
        imgs = [sorted(make_dataset(dirs[0], dirs[1])) for dirs in roots]
        lengths = [len(img) for img in imgs]

        if 0 in lengths:
            raise(RuntimeError("Found 0 images in: " + letters[lengths.index(0)] + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.data_root = data_root
        self.mode = mode
        self.roots = roots
        self.imgs = imgs

        self.letters = letters
        self.n_datasets = n_datasets
        self.label_use = [int(l) > 0 for l in label_use.split('|')]
        self.return_paths = return_paths

        self.resize_to = resize_to
        self.crop_to = crop_to

    def load_samples(self, n_samples, d_index):

        sample_list = [self.load_preprocess(
            self.imgs[d_index][i][0], self.imgs[d_index][i][1] if self.label_use[d_index] else None) for i in range(n_samples)]

        img_list = [s[0] for s in sample_list]
        msk_list = [s[1] for s in sample_list]

        return img_list, msk_list

    def load_preprocess(self, img_path, msk_path):

        # Loading.
        img = io.imread(img_path)
        msk = None
        if msk_path is not None:
            msk = io.imread(msk_path)
            # Resizing msk
            msk = transform.resize(msk, self.resize_to,
                                   order=0, preserve_range=True)
            msk[msk > 0] = 1
        else:
            msk = np.full(self.resize_to, fill_value=-1)
            
        # Random negative image
        if np.random.choice([0, 1], p=[0.5, 0.5]) and self.mode == 'train':
            img = util.invert(img)

        # Resizing.
        img = transform.resize(img, self.resize_to,
                               order=1, preserve_range=True)

        # Random crop.
        if self.resize_to[0] != self.crop_to[0] or self.resize_to[1] != self.crop_to[1]:

            pos = (np.random.randint(self.resize_to[0] - self.crop_to[0]),
                   np.random.randint(self.resize_to[1] - self.crop_to[1]))

            img = img[pos[0]:pos[0] + self.crop_to[0],
                      pos[1]:pos[1] + self.crop_to[1]]
            msk = msk[pos[0]:pos[0] + self.crop_to[0],
                      pos[1]:pos[1] + self.crop_to[1]]

        # Normalization and transformation to tensor.
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) - 0.5
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)

        msk = msk.astype(np.int64)
        msk = torch.from_numpy(msk)

        return img, msk

    def __getitem__(self, index):

        # Randomly choosing dataset.
        t = time.time()
        seed = int((t - int(t)) * 100) * index
        np.random.seed(seed)

#         perm = np.random.permutation(len(self.imgs))[:2]
        perm = np.random.randint(len(self.imgs), size=2)

        ind_a = perm[0]
        ind_b = perm[1]

        # Label usage.
        label_a = self.label_use[ind_a]
        label_b = self.label_use[ind_b]

        # Computing paths.
        img_path_a, msk_path_a, img_path_b, msk_path_b = None, None, None, None
        if label_a:
            img_path_a, msk_path_a = self.imgs[ind_a][index]
        else:
            img_path_a, _ = self.imgs[ind_a][index]
        if label_b:
            img_path_b, msk_path_b = self.imgs[ind_b][index]
        else:
            img_path_b, _ = self.imgs[ind_b][index]

        # Load function.
        img_a, msk_a = self.load_preprocess(img_path_a, msk_path_a)
        img_b, msk_b = self.load_preprocess(img_path_b, msk_path_b)

        if self.return_paths:
            return img_a, img_b, msk_a, msk_b, ind_a, ind_b, label_a, label_b, img_path_a.split('/')[-1], img_path_b.split('/')[-1]
        else:
            return img_a, img_b, msk_a, msk_b, ind_a, ind_b, label_a, label_b

    def __len__(self):

        return min([len(img) for img in self.imgs])


class ValImageFolder(data.Dataset):

    def __init__(self, data_root, mode, letter, resize_to=(284, 284), crop_to=(256, 256), return_paths=False):

        # letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        roots = (os.path.join(data_root, '%s%s' % (mode, letter)),
                 os.path.join(data_root, 'label_%s%s' % (mode, letter)))
        imgs = sorted(make_dataset(roots[0], roots[1]))
        length = len(imgs)

        if length == 0:
            raise(RuntimeError("Found 0 images in: " + letter + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.data_root = data_root
        self.mode = mode
        self.roots = roots
        self.imgs = imgs

        self.letter = letter
        self.label_use = True
        self.return_paths = return_paths

        self.resize_to = (256, 256)
        self.crop_to = crop_to

    def load_samples(self, n_samples, d_index):

        sample_list = [self.load_preprocess(
            self.imgs[d_index][i][0], self.imgs[d_index][i][1] if self.label_use[d_index] else None) for i in range(n_samples)]

        img_list = [s[0] for s in sample_list]
        msk_list = [s[1] for s in sample_list]

        return img_list, msk_list

    def load_preprocess(self, img_path, msk_path):

        # Loading.
        img = io.imread(img_path)
        msk = None
        if msk_path is not None:
            msk = io.imread(msk_path)
            # Resizing msk
            msk = transform.resize(msk, self.resize_to,
                                   order=0, preserve_range=True)
            msk[msk > 0] = 1
        else:
            msk = np.full(self.resize_to, fill_value=-1)

        # Resizing.
        img = transform.resize(img, self.resize_to,
                               order=1, preserve_range=True)

        # Random crop.
        # if self.resize_to[0] != self.crop_to[0] or self.resize_to[1] != self.crop_to[1]:

        #     pos = (np.random.randint(self.resize_to[0] - self.crop_to[0]),
        #            np.random.randint(self.resize_to[1] - self.crop_to[1]))

        #     img = img[pos[0]:pos[0] + self.crop_to[0],
        #               pos[1]:pos[1] + self.crop_to[1]]
        #     msk = msk[pos[0]:pos[0] + self.crop_to[0],
        #               pos[1]:pos[1] + self.crop_to[1]]

        # Normalization and transformation to tensor.
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) - 0.5
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)

        msk = msk.astype(np.int64)
        msk = torch.from_numpy(msk)

        return img, msk

    def __getitem__(self, index):

        # Computing paths.
        img_path_a, msk_path_a = None, None

        if self.label_use:
            img_path_a, msk_path_a = self.imgs[index]
        else:
            img_path_a, _ = self.imgs[index]
        # Load function.
        img_a, msk_a = self.load_preprocess(img_path_a, msk_path_a)

        if self.return_paths:
            return img_a, msk_a, img_path_a.split('/')[-1]
        else:
            return img_a, msk_a,

    def __len__(self):

        return len(self.imgs)
