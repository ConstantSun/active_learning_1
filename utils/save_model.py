import torch
from pathlib import Path


def save_model(net: torch.nn, optimizer, epoch, phase, dir_checkpoint, ckpt_name='ckpt_no_dropout.pth'):
    """
    Save the model to disk.

    Args:
        net: (todo): write your description
        torch: (todo): write your description
        nn: (todo): write your description
        optimizer: (todo): write your description
        epoch: (int): write your description
        phase: (todo): write your description
        dir_checkpoint: (str): write your description
        ckpt_name: (str): write your description
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'phase': phase
    }, dir_checkpoint + ckpt_name)


def load_model(net: torch.nn, dir_checkpoint, device):
    """
    Load a model from disk.

    Args:
        net: (todo): write your description
        torch: (todo): write your description
        nn: (str): write your description
        dir_checkpoint: (str): write your description
        device: (str): write your description
    """
    dropout_flag = "dropout" + str(net.is_dropout)
    interrupted_ckpt_dir = Path(dir_checkpoint + f'{dropout_flag}_ckpt.pth')

    if interrupted_ckpt_dir.is_file():
        checkpoint = torch.load(interrupted_ckpt_dir)
        net.load_state_dict(
            torch.load(checkpoint['model_state_dict']), map_location=device
        )
        pre_phase = checkpoint['phase']
        pre_epoch = checkpoint['epoch']
    else:
        pre_phase = 0
        pre_epoch = 0
    return net, pre_phase, pre_epoch
