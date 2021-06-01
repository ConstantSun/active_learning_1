from os.path import splitext
from os import listdir
import sys
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


class RestrictedDataset(Dataset):
    def __init__(self, imgs_dir: str, masks_dir: str, selected_id_images: list, train=True):
        """
        imgs_dir: image directory
        masks_dir: mask directory
        selected_id_images: list of image names in data pooling (the selected images)
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.train = train
        if train:
            # Active learning:
            # Updating selected image for the next training phase:
            # # self.ids: a list of images name, e.g:  [GEMS_IMG__2010_MAR__12__HA122541__F8HB4A50_24,...]
            self.ids = selected_id_images
        else:
            self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                        if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        """
        Returns the length of the length.

        Args:
            self: (todo): write your description
        """
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        """
        Preprocess a pil image.

        Args:
            cls: (todo): write your description
            pil_img: (todo): write your description
        """
        pil_img = pil_img.resize((256, 256))
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255.0

        return img_trans


    @classmethod
    def postprocess_augment(cls, img_nd: np.array):
        """
        Postprocess an image.

        Args:
            cls: (todo): write your description
            img_nd: (todo): write your description
            np: (todo): write your description
            array: (array): write your description
        """
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255.0

        return img_trans

    def strong_aug(self, p=0.5):
        """
        Return a 3darray of the quadrature.

        Args:
            self: (todo): write your description
            p: (todo): write your description
        """
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=p)

    def __getitem__(self, i):
        """
        Get the mask of - image.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        idx = self.ids[i]

        mask_file = glob(self.masks_dir + idx  + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        if self.train == False:
            img = self.preprocess(img)
            mask = self.preprocess(mask)

            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'id': self.ids[i]
            }
        else:
            img, mask = img.resize((256, 256)), mask.resize((256, 256))
            assert img.size == mask.size, \
                f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

            img_nd, mask = np.array(img), np.array(mask)
            augmentation = self.strong_aug(p=0.6)
            augmented = augmentation(image=img_nd, mask=mask) 
            a_img = augmented["image"]
            a_mask = augmented["mask"]
            a_img, a_mask = self.postprocess_augment(a_img), self.postprocess_augment(a_mask)
            return {
                'image': torch.from_numpy(a_img).type(torch.FloatTensor),
                'mask': torch.from_numpy(a_mask).type(torch.FloatTensor),
                'id': self.ids[i]
            }
