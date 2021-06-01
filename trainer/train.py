from tqdm import tqdm
import torch
import torch.nn as nn
from trainer.eval import eval_net
from pathlib import Path
import os


# trong mỗi phase, lưu lại best model để làm training cho phase tiếp theo.
def train(net: torch.nn, data_train, train_loader, criterion, optimizer, writer, epochs, pre_epoch, n_channels,
          device, global_step, test_loader, n_classes, dir_checkpoint, logging, phase):
    """
    Training function.

    Args:
        net: (todo): write your description
        torch: (todo): write your description
        nn: (array): write your description
        data_train: (array): write your description
        train_loader: (todo): write your description
        criterion: (int): write your description
        optimizer: (todo): write your description
        writer: (todo): write your description
        epochs: (array): write your description
        pre_epoch: (bool): write your description
        n_channels: (int): write your description
        device: (todo): write your description
        global_step: (todo): write your description
        test_loader: (todo): write your description
        n_classes: (int): write your description
        dir_checkpoint: (str): write your description
        logging: (todo): write your description
        phase: (todo): write your description
    """
    best_test_iou_score = 0
    dropout_flag = "dropout" + str(net.is_dropout)
    right_previous_ckpt_dir = Path(dir_checkpoint + f'best_CP_one64th_{dropout_flag}.pth')
    if right_previous_ckpt_dir.is_file():
        net.load_state_dict(
            torch.load(dir_checkpoint + f'best_CP_one64th_{dropout_flag}.pth', map_location=device)
        )
    # epoch_start = pre_epoch if pre_epoch != 0 else 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        n_train = len(data_train)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if n_classes == 1 else torch.long

                true_masks = true_masks.to(device=device, dtype=mask_type)
                # print("img shape: ", imgs.shape) # 4 3 256 256
                masks_pred = net(imgs)  # return BCHW = 4_1_256_256

                # print("mask gen shape: ", masks_pred.shape)
                _tem = net(imgs)
                # print("IS DIFFERENT OR NOT: ", torch.sum(masks_pred - _tem))
                true_masks = true_masks[:, :1, :, :]
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                # writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1
        # Tính dice và iou score trên tập Test set, ghi vào tensorboard .
        test_score_dice, test_score_iou = eval_net(net, test_loader, n_classes, device)
        if test_score_iou > best_test_iou_score:
            best_test_iou_score = test_score_iou
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'best_CP_one64th_{dropout_flag}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        logging.info('Test Dice Coeff: {}'.format(test_score_dice))
        print('Test Dice Coeff: {}'.format(test_score_dice))
        writer.add_scalar(f'Phase_{phase}_Dice_{dropout_flag}/test', test_score_dice, epoch)

        logging.info('Test IOU : {}'.format(test_score_iou))
        print('Test IOU : {}'.format(test_score_iou))
        writer.add_scalar(f'Phase_{phase}_IOU_{dropout_flag}/test', test_score_iou, epoch)

    print(f"Phase_{phase}_best iou: ", best_test_iou_score)
    # torch.save(net.state_dict(),
    #            dir_checkpoint + 'ckpt.pth')
    writer.add_scalar(f'Phases_IOU_{dropout_flag}/test', best_test_iou_score, phase)
    return pre_epoch
