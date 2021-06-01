""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import logging

from PIL import Image

from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        """
        Initialize the parser.

        Args:
            self: (todo): write your description
            root: (str): write your description
            parser: (todo): write your description
            class_map: (todo): write your description
            load_bytes: (todo): write your description
            transform: (str): write your description
        """
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        """
        Retrieve an item from index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        """
        Returns the length of the parser.

        Args:
            self: (todo): write your description
        """
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        """
        Return the filename of the filename.

        Args:
            self: (todo): write your description
            index: (str): write your description
            basename: (str): write your description
            absolute: (str): write your description
        """
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        """
        Return a list of filenames.

        Args:
            self: (todo): write your description
            basename: (str): write your description
            absolute: (todo): write your description
        """
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        """
        Initialize the parser.

        Args:
            self: (todo): write your description
            root: (str): write your description
            parser: (todo): write your description
            split: (int): write your description
            is_training: (bool): write your description
            batch_size: (int): write your description
            class_map: (todo): write your description
            load_bytes: (todo): write your description
            transform: (str): write your description
        """
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training, batch_size=batch_size)
        else:
            self.parser = parser
        self.transform = transform
        self._consecutive_errors = 0

    def __iter__(self):
        """
        Iterate over an iterator.

        Args:
            self: (todo): write your description
        """
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            yield img, target

    def __len__(self):
        """
        Returns the length of the parser.

        Args:
            self: (todo): write your description
        """
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        """
        Sets the filename

        Args:
            self: (todo): write your description
            index: (str): write your description
            basename: (str): write your description
            absolute: (str): write your description
        """
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        """
        Return a list of filenames.

        Args:
            self: (todo): write your description
            basename: (str): write your description
            absolute: (todo): write your description
        """
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            num_splits: (int): write your description
        """
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        """
        Set the transformation of the dataset.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        """
        Return the transformed dataset.

        Args:
            self: (array): write your description
        """
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        """
        Transform x to an array.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        self._set_transforms(x)

    def _normalize(self, x):
        """
        Normalize x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        """
        Return a tuple of items.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        """
        Returns the number of the dataset.

        Args:
            self: (todo): write your description
        """
        return len(self.dataset)
