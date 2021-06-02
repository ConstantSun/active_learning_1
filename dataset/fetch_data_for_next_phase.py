import torch
from tqdm import tqdm
from dataset.dynamic_dataloader import RestrictedDataset
from torch.utils.data import DataLoader

import json
from utils import acquisition_function
import random

from trainer.setting import dir_img, dir_mask, GAUSS_ITERATION, NUM_FETCHED_IMG


def add_image_id_to_pool(id: str, filename="pooling_data.json"):
    """id: image name, e.g: GEMS_IMG__2010_MAR__12__HA122541__F8HB4A50_24"""
    with open(filename, 'r+') as f:
        dic = json.load(f)
        dic["ids"].append(id)
    with open(filename, 'w') as file:
        json.dump(dic, file)


def delete_image_id_from_pool(id: str, filename="pooling_data.json"):
    """id: image name, e.g: GEMS_IMG__2010_MAR__12__HA122541__F8HB4A50_24"""
    with open(filename, 'r+') as f:
        dic = json.load(f)
        dic["ids"].remove(id)
    with open(filename, 'w') as file:
        json.dump(dic, file)


def get_pool_data(filename="pooling_data.json"):
    """return a list of image names (image id)"""
    with open(filename, 'r+') as f:
        dic = json.load(f)
        return dic["ids"]


def update_training_pool_ids(net: torch.nn, training_pool_ids_path: str, all_training_data,
                               device: str, acquisition_func: str = "cfe"):
    """
    training_pool_ids_path: the path to json file which contains images id in training pool.
    acquisition_func: string name of acquisition function:
                    available function: mutual_information, mean_first_entropy, category_first_entropy, std, random
    This function will use an acquisition function to collect new NUM_FETCHED_IMG imgs into training pool each phase.
        /Increase the json file NUM_FETCHED_IMG more imgs each phase.
    """
    batch_size = 1
    training_pool_data = get_pool_data(training_pool_ids_path)
    all_training_data = get_pool_data(all_training_data)
    active_pool = set(all_training_data) - set(training_pool_data)

    if acquisition_func == "cfe":
        evaluation_criteria = acquisition_function.category_first_entropy
    elif acquisition_func == "mfe":
        evaluation_criteria = acquisition_function.mean_first_entropy
    elif acquisition_func == "mi":
        evaluation_criteria = acquisition_function.mutual_information
    elif acquisition_func == "std":
        evaluation_criteria = acquisition_function.std
    elif acquisition_func == "random":
        random_elements = random.sample(active_pool, NUM_FETCHED_IMG)
        for i in random_elements:
            add_image_id_to_pool(i, training_pool_ids_path)
        return
    else:
        print("Error choosing acquisition function")
        evaluation_criteria = None

    dataset = RestrictedDataset(dir_img, dir_mask, list(active_pool), train=False, active=True)
    pool_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    value = []
    imgs_id = []

    net.eval()
    n_pool = len(dataset)
    with tqdm(total=n_pool, desc='STD calculating', unit='batch', leave=False) as pbar:
        for ind, batch in enumerate(tqdm(pool_loader)):
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            _value = evaluation_criteria(GAUSS_ITERATION, net, imgs)
            _imgs_id = batch['id']
            for i in range(batch_size):
                if i >= len(_value):
                    continue
                value.extend(_value)
                imgs_id.extend(_imgs_id)
            pbar.update()
    value, imgs_id = zip(*sorted(zip(value, imgs_id))) # order = ascending
    print("length of value/imgs_id: ", len(value), len(imgs_id))
    top_Kimg = imgs_id[-NUM_FETCHED_IMG:] # the higher
    for i in top_Kimg:
        add_image_id_to_pool(i, training_pool_ids_path)
    print("Adding successfully!")
