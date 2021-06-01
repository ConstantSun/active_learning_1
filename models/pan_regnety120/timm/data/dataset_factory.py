import os

from .dataset import IterableImageDataset, ImageDataset


def _search_split(root, split):
    """
    Search for a file path into the root path.

    Args:
        root: (str): write your description
        split: (str): write your description
    """
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    """
    Create a dataset.

    Args:
        name: (str): write your description
        root: (str): write your description
        split: (str): write your description
        search_split: (str): write your description
        is_training: (bool): write your description
        batch_size: (int): write your description
    """
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds
