import math
import torch.nn.functional as F

def compute_patched_disc_loss(net_disc, image, disc_args, args):
    """
    Compute the discriminator loss over a complete image and its patches.

    Parameters:
      net_disc: the discriminator network.
      image: input tensor of shape (B, C, H, W).
      disc_args: dictionary of additional keyword arguments for net_disc 
                 (e.g. {"for_G": True} or {"for_real": True}).
      args: training arguments; must contain:
            - num_patches: number of patches per side.
            - lambda_disc_complete: weighting for the complete image loss.
    Returns:
      A scalar tensor representing the combined loss.
    """
    max_pixels = 224
    B, C, H, W = image.shape
    factor = math.ceil(H / max_pixels)
    if factor > args.num_patches:
        factor = args.num_patches
    new_size = max_pixels * factor
    # Resize the image to a size divisible into patches
    image_resized = F.interpolate(image, size=(new_size, new_size), mode='bilinear', align_corners=False)
    # Compute loss on the complete (resized) image
    complete_loss = net_disc(image_resized, **disc_args).mean()
    if factor <= 1:
        return complete_loss
    patch_losses = []
    patch_size = max_pixels
    # Divide into patches and compute the loss on each patch.
    for i in range(factor):
        for j in range(factor):
            patch = image_resized[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch_loss = net_disc(patch, **disc_args).mean()
            patch_losses.append(patch_loss)
    # Combine: the complete image loss is weighted by lambda_disc_complete,
    # and the patch losses are averaged with a denominator of (num_patches+1)
    final_loss = (args.lambda_disc_complete * complete_loss + sum(patch_losses)) / (len(patch_losses) + args.lambda_disc_complete)
    return final_loss

def compute_patched_lpips_loss(net_lpips, image_pred, image_target, args):
    """
    Compute the LPIPS loss over a complete image and its patches.

    Parameters:
      net_lpips: the LPIPS network.
      image_pred: predicted image tensor of shape (B, C, H, W).
      image_target: ground truth image tensor of shape (B, C, H, W).
      args: training arguments; must contain:
            - num_patches: number of patches per side.
            - lambda_lpips_complete: weighting for the complete image loss.
    Returns:
      A scalar tensor representing the combined LPIPS loss.
    """
    max_pixels = 512
    B, C, H, W = image_pred.shape
    factor = math.ceil(H / max_pixels)
    if factor > args.num_patches:
        factor = args.num_patches
    new_size = max_pixels * factor
    # Resize both images to the new size.
    pred_resized = F.interpolate(image_pred, size=(new_size, new_size), mode='bilinear', align_corners=False)
    target_resized = F.interpolate(image_target, size=(new_size, new_size), mode='bilinear', align_corners=False)
    # Compute LPIPS on the complete image.
    complete_loss = net_lpips(pred_resized, target_resized).mean()
    if factor <= 1:
        return complete_loss
    patch_losses = []
    patch_size = max_pixels
    # Process each patch.
    for i in range(factor):
        for j in range(factor):
            pred_patch = pred_resized[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            target_patch = target_resized[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch_loss = net_lpips(pred_patch, target_patch).mean()
            patch_losses.append(patch_loss)
    final_loss = (args.lambda_lpips_complete * complete_loss + sum(patch_losses)) / (len(patch_losses) + args.lambda_lpips_complete)
    return final_loss
