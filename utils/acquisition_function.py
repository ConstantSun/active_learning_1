import torch


def get_entropy_in_a_prob_mask(mask: torch.tensor):
    """
    mask: batch x (c) x h x w.
    return shape: batch.
    """
    batch = mask.shape[0]
    flattened = mask.reshape(batch, -1)
    res = flattened*torch.log(flattened) + (1-flattened)*torch.log(1-flattened)
    res = res.sum(dim=1)
    return res


def category_first_entropy(GAUSS_ITERATION: int, net: torch.nn, imgs: torch.tensor) -> torch.tensor:
    """
    input:
    imgs: batch of imgs, shape: batchx(c)xhxw.
    output:
    category first entropy of that batch, shape: batch.

    This query function is calculating the entropy between all the classes of one
    pixel first, then average it with multiple models.
    The higher result the higher uncertainty.
    """
    entropy_list = []
    for i in range(GAUSS_ITERATION):
        with torch.no_grad():
            logits = net(imgs)
            y_pred = torch.sigmoid(logits)
            y_pred = y_pred[:, :1, :, :] # batch x 1 x hxw
            entropy_list.append(get_entropy_in_a_prob_mask(y_pred)) # shape: ind, batch
    res = torch.stack(entropy_list, dim=0)
    res = res.mean(dim=1)
    return res


def mean_first_entropy(GAUSS_ITERATION: int, net: torch.nn, imgs: torch.tensor) -> torch.tensor:
    """
    input:
    imgs: batch of imgs, shape: batchx(c)xhxw.
    output:
    mean_first_entropy of that batch, shape: batch.

    This query function is extracting mean of probability from multiple models
    first, then calculating the entropy based on the output.
    The higher result the higher uncertainty.

    """
    y_pred_samples = []
    for i in range(GAUSS_ITERATION):
        with torch.no_grad():
            logits = net(imgs)
            y_pred = torch.sigmoid(logits)
            y_pred = y_pred[:, :1, :, :]
            y_pred_samples.append(y_pred[:, 0, :, :])  # y_pred_samples's shape: (inx, bat, H, W )
    y_pred_samples = torch.stack(y_pred_samples, dim=0)
    y_pred_samples = y_pred_samples.type(torch.cuda.FloatTensor)
    mean_y_pred = y_pred_samples.mean(dim=0)  # shape: batch, H, W
    res = get_entropy_in_a_prob_mask(mean_y_pred)
    return res


def mutual_information(GAUSS_ITERATION: int, net: torch.nn, imgs: torch.tensor) -> torch.tensor:
    """
    This query function calculates the difference of two entropy calculated above.
    H_mean − H_cato

    """
    res = mean_first_entropy(GAUSS_ITERATION, net, imgs) - category_first_entropy(GAUSS_ITERATION, net, imgs)
    return res


def get_segmentation_mask_uncertainty(gened_std_mask: torch.tensor) -> torch.tensor:
    """
    gened_std_mask, Shape: batchx(channel)xHxW - ex: 4x(1)xHxW : generated std mask, by taking std of output masks when using MC dropout.
    return: a list of sum of std value of a image in n=batch images.

    """
    # flattening mask
    # print("gened mask shape: ", gened_mask.shape)
    # print("gt mask shape : ", gt_mask.shape)
    batch = gened_std_mask.shape[0]
    flattened_gened_mask = gened_std_mask.reshape(batch, -1)
    return gened_std_mask.sum(dim=1).tolist()[0]


def std(GAUSS_ITERATION: int, net: torch.nn, imgs: torch.tensor) -> torch.tensor:
    """
    Applies the standard deviation.

    Args:
        GAUSS_ITERATION: (int): write your description
        net: (todo): write your description
        torch: (todo): write your description
        nn: (array): write your description
        imgs: (array): write your description
        torch: (todo): write your description
        tensor: (todo): write your description
    """
    y_pred_samples = []
    for i in range(GAUSS_ITERATION):
        with torch.no_grad():
            logits = net(imgs)
            y_pred = torch.sigmoid(logits)
            # y_pred = (y_pred > 0.5).float()
            y_pred = y_pred[:, :1, :, :]
            y_pred_samples.append(y_pred[:, 0, :, :])  # y_pred_samples's shape: (inx, bat, H, W )
    y_pred_samples = torch.stack(y_pred_samples, dim=0)
    y_pred_samples = y_pred_samples.type(torch.FloatTensor)
    mean_y_pred = y_pred_samples.mean(dim=0)  # shape: batch, H, W
    std_y_pred = y_pred_samples.std(dim=0)  # shape: batch, H, W
    _std = get_segmentation_mask_uncertainty(std_y_pred)
