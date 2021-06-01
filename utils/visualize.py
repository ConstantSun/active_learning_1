# import torch
# import cv2
# import numpy as np
# import logging
# GAUSS_ITERATION = 100
#
#
# def visualize_to_tensorboard(test_loader, train_loader_un_shuffle, writer, device,
#                              net, n_channels, n_classes, batch_size, epoch):
#     for batch_idx, (batch_test, batch_train) in enumerate(
#             zip(test_loader, train_loader_un_shuffle)):
#         if batch_idx > 3:  # take 4 batches for visualizing.
#             break
#
#         # visualize the first batch in test, val, train set.
#         for set_idx, batch in enumerate([batch_test, batch_train]):
#             imgs = batch['image']
#             y_true = batch['mask']
#
#             assert imgs.shape[1] == n_channels, \
#                 f'Network has been defined with {n_channels} input channels, ' \
#                 f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
#                 'the images are loaded correctly.'
#
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             mask_type = torch.float32 if n_classes == 1 else torch.long
#             y_true = y_true.to(device=device, dtype=mask_type)
#             y_pred_samples = []
#             for i in range(GAUSS_ITERATION):
#                 with torch.no_grad():
#                     logits = net(imgs)
#                     y_pred = torch.sigmoid(logits)
#                     y_pred = (y_pred > 0.5).float()
#                     y_pred = y_pred[:, :1, :, :]
#                     y_pred_samples.append(y_pred[:, 0, :, :])  # y_pred_samples's shape: (inx, bat, H, W )
#
#             y_pred_samples = torch.stack(y_pred_samples, dim=0)
#             y_pred_samples = y_pred_samples.type(torch.FloatTensor)
#             mean_y_pred = y_pred_samples.mean(dim=0)  # shape: batch, H, W
#             var_y_pred = y_pred_samples.var(dim=0)  # shape: batch, H, w
#
#             mean_y_pred = mean_y_pred.detach().cpu()
#             var_y_pred = var_y_pred.detach().cpu()
#             y_true = y_true[:, 0, :, :].detach().cpu()  # --- b,h,w
#             print("Max of var y: ", var_y_pred[0].max())
#
#             _total_list = []
#             for inx in range(batch_size):
#                 yellow = (255, 255, 0)  # don't change
#                 red = (255, 0, 0)  # don't change
#                 blue = (0, 0, 255)  # don't change
#
#                 _var_y_pred = torch.tensor([var_y_pred[inx].numpy() * yellow[i] for i in range(3)])  # c,h,w
#                 _min = _var_y_pred.min()
#                 _max = _var_y_pred.max()
#                 _var_y_pred = (_var_y_pred - _min) / (_max - _min) * 255
#
#                 _y_true = y_true[inx]  # h,w
#                 logging.info(f"_y_true  type: {type(_y_true)}, shape: {_y_true.shape}")
#                 _y_true = (_y_true > 0).float()
#                 # logging.info(f"_y_true: {_y_true.shape} - {type(_y_true)}")
#                 _y_true = _y_true.numpy()
#                 ero = cv2.erode(np.uint8(_y_true), np.ones((5, 5), dtype=np.uint8))
#                 _bound_true = _y_true - ero
#                 _bound_true = torch.tensor([_bound_true * red[i] for i in range(3)])  # c,h,w
#
#                 mean_y_pred = mean_y_pred[inx]
#                 logging.info(f"mean_y_pred type : {type(mean_y_pred)}, shape: {mean_y_pred.shape}")
#
#                 if isinstance(mean_y_pred, torch.Tensor):
#                     mean_y_pred = (mean_y_pred > 0.5).float()
#                 elif isinstance(mean_y_pred, np.float32):
#                     mean_y_pred = torch.Tensor([mean_y_pred])
#                 else:
#                     mean_y_pred = torch.Tensor(mean_y_pred)
#
#                 # logging.info(f'mean_y_pred: {mean_y_pred.shape} - {type(mean_y_pred)}')
#                 mean_y_pred = mean_y_pred.numpy()
#                 ero = cv2.erode(np.uint8(mean_y_pred), np.ones((5, 5), dtype=np.uint8))
#                 _bound_mean = mean_y_pred - ero
#                 _bound_mean = torch.tensor([_bound_mean * blue[i] for i in range(3)])  # c,h,w
#
#                 _bound_mean = _bound_mean.type(torch.LongTensor)
#                 _var_y_pred = _var_y_pred.type(torch.LongTensor)
#                 _bound_true = _bound_true.type(torch.LongTensor)
#
#                 _total = torch.where(_bound_mean[2] != 0, _bound_mean,
#                                      _var_y_pred)  # vi bound mean mau blue nen xet channel inx 2
#                 _total = torch.where(_bound_true[0] != 0, _bound_true,
#                                      _total)  # vi bound true mau red nen xet channel inx 0
#
#                 _img = (imgs[inx].cpu() * 255).type(torch.LongTensor)  # convert img to original value (c,h,w)
#                 _total = torch.where(_total != 0, _total, _img)
#                 _total = torch.cat((_total, _img), dim=2)  # adding input image to the right of the result image
#                 _total_list.append(_total.type(torch.uint8))
#
#             _total_list = torch.stack(_total_list, dim=0)
#
#             if set_idx == 0:
#                 writer.add_images(f'test/blue_mean___yellow_var_pred___red_GT_{inx}_{batch_idx}', _total_list, epoch)
#             elif set_idx == 1:
#                 writer.add_images(f'train/blue_mean___yellow_var_pred___red_GT_{inx}_{batch_idx}', _total_list, epoch)
#             # elif set_idx == 2:
#             #     writer.add_images(f'val/blue_mean___yellow_var_pred___red_GT_{inx}_{batch_idx}', _total_list, epoch)
#
