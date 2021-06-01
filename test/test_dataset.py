from torch.utils.data import DataLoader
from dataset.dynamic_dataloader import RestrictedDataset
from dataset.fetch_data_for_next_phase import get_pool_data

# ailab
dir_img = '/dataset.local/all/hangd/src_code_3/Pytorch-UNet/dataset/imgs/'
dir_mask = '/dataset.local/all/hangd/src_code_3/Pytorch-UNet/dataset/masks/'

dir_img_test = '/dataset.local/all/hangd/src_code_3/Pytorch-UNet/data_test/imgs/'
dir_mask_test = '/dataset.local/all/hangd/src_code_3/Pytorch-UNet/data_test/masks/'

dir_img_draft = 'dataset/hangd/cardi/RobustSegmentation/data_draft/imgs/'
dir_mask_draft = 'dataset/hangd/cardi/RobustSegmentation/data_draft/masks/'

pool_data = get_pool_data()
dataset = RestrictedDataset(dir_img, dir_mask, pool_data, True)
pool_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

batch = next(iter(pool_loader))
img = batch['image']
mask = batch['mask']
id = batch['id']
print("train_loader: ", img.shape, mask.shape)
print("id: ", id)
