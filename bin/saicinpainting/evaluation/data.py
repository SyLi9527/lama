import glob
import os

import cv2
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np
from io import BytesIO
import base64

from torch.utils.data import Dataset
import torch.nn.functional as F

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255

    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img

def transform_b64_to_img(b64_string, mode='RGB', return_orig=False):
    # mat need antialias method
    img_data= Image.open(BytesIO(base64.b64decode(b64_string)))

    img_data = img_data.convert(mode)

    if mode == 'L':
        img_data = ImageOps.invert(img_data)

    img = np.array(img_data)

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

def refine_mask(b64_image, b64_mask):
    # b64_img -> rgb -> np array
    img_data= Image.open(BytesIO(base64.b64decode(b64_image)))
    img_data = img_data.convert('RGB')
    img = np.array(img_data)

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255

    mask_data= Image.open(BytesIO(base64.b64decode(b64_mask)))
    mask_data = mask_data.convert('L')
    # mask_data = ImageOps.invert(mask_data)

    img_width, img_height = img_data.size

    mask_data = mask_data.resize((max(img_width, img_height), max(img_width, img_height)), Image.ANTIALIAS)

    mask = np.array(mask_data)

    if mask.ndim == 3:
        mask = np.transpose(mask, (2, 0, 1))
    out_mask = mask.astype('float32') / 255


    # crop mask to be the same size as the image
    mask_left = int((max(img_width, img_height) - img_width) / 2)
    mask_top = int((max(img_width, img_height) - img_height) / 2)
    mask_right = mask_left + img_width
    mask_bottom = mask_top + img_height
    out_mask = out_mask[mask_top:mask_bottom, mask_left:mask_right]

    return out_img, out_mask

class b64InpaintingDataset(Dataset):
    def __init__(self, b64_image, b64_mask, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.b64_image = b64_image
        self.b64_mask = b64_mask
        # self.img_suffix = img_suffix
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return 1

    def __getitem__(self, i):
        # image = transform_b64_to_img(self.b64_image, mode='RGB')
        # unrefined_mask = transform_b64_to_img(self.b64_mask, mode='L')
        image, refined_mask = refine_mask(self.b64_image, self.b64_mask)
        result = dict(image=image, mask=refined_mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor
    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class OurInpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, 'mask', '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [os.path.join(self.datadir, 'img', os.path.basename(fname.rsplit('-', 1)[0].rsplit('_', 1)[0]) + '.png') for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        result = dict(image=load_image(self.img_filenames[i], mode='RGB'),
                      mask=load_image(self.mask_filenames[i], mode='L')[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class PrecomputedInpaintingResultsDataset(InpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix='_inpainted.jpg', **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = load_image(self.pred_filenames[i])
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result

class OurPrecomputedInpaintingResultsDataset(OurInpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix="png", **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.basename(os.path.splitext(fname)[0]) + f'_inpainted.{inpainted_suffix}')
                               for fname in self.mask_filenames]
        # self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
        #                        for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = self.file_loader(self.pred_filenames[i])

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result

class InpaintingEvalOnlineDataset(Dataset):
    def __init__(self, indir, mask_generator, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None,  **kwargs):
        self.indir = indir
        self.mask_generator = mask_generator
        self.img_filenames = sorted(list(glob.glob(os.path.join(self.indir, '**', f'*{img_suffix}' ), recursive=True)))
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, i):
        img, raw_image = load_image(self.img_filenames[i], mode='RGB', return_orig=True)
        mask = self.mask_generator(img, raw_image=raw_image)
        result = dict(image=img, mask=mask)

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
        return result