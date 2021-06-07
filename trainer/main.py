import os
import sys
# sys.path.append("/home/cotai/Phi/active/cleancode/")
# sys.path.append("/content/active_learning_1/") #colab
sys.path.append(os.path.dirname(os.getcwd()))


from models.pan_regnety120.pan import PAN
import argparse
import logging


import torch
import torch.nn as nn
from torch import optim

from dataset.fetch_data_for_next_phase import get_pool_data, update_training_pool_ids
from trainer.train import train

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.dynamic_dataloader import RestrictedDataset
from pathlib import Path
from utils.save_model import save_model, load_model
from trainer.setting import dir_img, dir_mask, dir_img_test, dir_mask_test, GAUSS_ITERATION, NUM_FETCHED_IMG

global val_iou_score
global best_val_iou_score
global best_test_iou_score


val_iou_score = 0.
best_val_iou_score = 0.
best_test_iou_score = 0.


def train_net(
        dir_checkpoint,
        n_classes,
        n_channels,
        device,
        epochs=30,
        save_cp=True,
        acquisition_function="random"):
    """
    Train the model.

    Args:
        dir_checkpoint: (bool): write your description
        n_classes: (int): write your description
        n_channels: (int): write your description
        device: (todo): write your description
        epochs: (todo): write your description
        save_cp: (bool): write your description
        acquisition_function: (todo): write your description
    """

    global best_val_iou_score
    global best_test_iou_score

    net = PAN(is_dropout=True)
    net.to(device=device)

    net_no_dropout = PAN(is_dropout=False)
    net_no_dropout.to(device=device)

    batch_size = 4
    lr = 1e-5
    writer = SummaryWriter(
        comment=f'_{net.__class__.__name__}_LR_{lr}_BS_{batch_size}_{acquisition_function}_ACQUISITION'
    )
    global_step = 0

    logging.basicConfig(filename=f"./logging_{acquisition_function}.txt",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)
    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer_no_dropout = optim.RMSprop(net_no_dropout.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler_no_dropout = optim.lr_scheduler.ReduceLROnPlateau(optimizer_no_dropout, 'min' if n_classes > 1 else 'max', patience=2)
    if n_classes > 1:
        criterion_no_dropout = nn.CrossEntropyLoss()
    else:
        criterion_no_dropout = nn.BCEWithLogitsLoss()

    num_phases = 50
    # total 2689 imgs, within each phase: fetching 100 imgs to training set.
    training_pool_ids_path =  f"{data_dir}/data_one64th_{acquisition_function}.json"
    all_training_data =  f"{data_dir}/data_all.json"

    pre_phase = 0
    pre_epoch = 0

    # khi kill process, cần load lại model tại phase x, epoch y , tên ckpt là dropouttrue_ckpt.pth
    # Load iterrupted model
    net, pre_phase, pre_epoch = load_model(net, dir_checkpoint, device)

    # Cuối  mỗi phase, load lại best model và tiếp tục training, lưu tên dưới dạng dropouttrue_ckpt_best.pth
    for phase in range(pre_phase, num_phases+1):
        # Within a phase, save the best epoch (having highest test_iou) checkpoint and save its test_iou to TF_Board
        #                 also, load the best right previous checkpoint
        selected_images = get_pool_data(training_pool_ids_path)
        data_train = RestrictedDataset(dir_img, dir_mask, selected_images, train=True)
        data_test = RestrictedDataset(imgs_dir=dir_img_test, masks_dir=dir_mask_test, selected_id_images=None, train=False)
        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=True)
        test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                                 drop_last=True)

        # Training with Dropout
        train(net, data_train, train_loader, criterion, optimizer, writer, epochs, pre_epoch, n_channels,
              device, global_step, test_loader, n_classes, dir_checkpoint, logging, phase)

        # Training without Dropout
        train(net_no_dropout, data_train, train_loader, criterion_no_dropout, optimizer_no_dropout, writer, epochs, pre_epoch, n_channels,
              device, global_step, test_loader, n_classes, dir_checkpoint, logging, phase)

        # Fetching data for next phase - Update pooling images.
        update_training_pool_ids(net, training_pool_ids_path, all_training_data, device, acquisition_func=acquisition_function)
    writer.close()


def get_args():
    """
    Parse command line arguments.

    Args:
    """
    parser = argparse.ArgumentParser(description='Train the Network on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-acq', '--acquisition_func', dest='acquisition_func', type=int, default=0,
                        help='Choose acquisition function index for collecting data')
    parser.add_argument('-cuda', '--cuda-inx', type=int, nargs='?', default=1,
                        help='index of cuda', dest='cuda_inx')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-savedir', '--savedir', dest='save_directory', type=str, default="/content/drive/MyDrive/thesis_uni/dlv3_dpn98",
                        help='Directory of tfboard and checkpoint path')
    parser.add_argument('-datadir', '--datadir', dest='data_directory', type=str, default="/content/drive/MyDrive/thesis_uni/act_1",
                        help='Directory of tfboard and checkpoint path')        
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()


def get_acquisition_func(i: int):
    """
    Input : {} function }

    Args:
        i: (todo): write your description
    """
    switcher = {
        0: "cfe",
        1: "mfe", #
        2: "std",
        3: "mi",
        4: "random", #
    }
    return switcher.get(i, "cfe")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    tf_dir = None                # not Colab
    current_dir = os.getcwd()    # not Colab
    data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "database")

    # current_dir = args.save_directory  # colab
    # tf_dir      = args.save_directory  # colab
    # data_dir    = args.data_directory  # colab
    

    try:
        os.mkdir(os.path.join(current_dir, "check_point_active"))
        logging.info('Created ckpt active directory')
    except OSError:
        pass

    acquisition_func = get_acquisition_func(args.acquisition_func)
    dir_ckp = os.path.join(current_dir, f"check_point_active/{acquisition_func}/")  
        

    if torch.cuda.is_available():
        _device = 'cuda:' + str(args.cuda_inx)
    else:
        _device = 'cpu'
    device = torch.device(_device)
    logging.info(f'Using device {device}')

    n_classes = 1
    n_channels = 3
    bilinear = True

    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\tUsing {acquisition_func} STRATEGY for collecting data for next phase \n'                 
                 f'\t{"Bilinear" if bilinear else "Transposed conv"} upscaling')

    # For a specific architecture
    try:
        train_net(dir_checkpoint=dir_ckp,
                  n_classes=n_classes,
                  n_channels=n_channels,
                  epochs=args.epochs,
                  device=device,
                  acquisition_function=acquisition_func,
                  tf_log_dir=tf_dir,
                  data_dir=data_dir
                  )
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
