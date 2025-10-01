from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb


def plot_image_grid(images: np.ndarray, plot_size: int = 3, save_dir: str = None, filename: str = None, return_figure=False):
    """
    Description:
        Given an array of images (N, H, W, 3), make an image grid (2, N/2) and save in .jpg format. 
    """
    if return_figure:
        assert save_dir is None and filename is None, "If return_figure is True, save_dir and filename should not be given."
    else:
        assert save_dir is not None and filename is not None, "If return_figure is False, save_dir and filename should be given"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # # Reshape images if needed
    # num_images = images.shape[0]
    # if num_images % 2 != 0:
    #     images = images[:num_images // 2 * 2] # Ensure even number of images
    # print("np.max", np.max(images))
    # print("np.min", np.min(images))
    
    # Make image grid
    num_rows = 2
    num_cols = images.shape[0] // 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(plot_size * num_cols, plot_size * num_rows))
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i, j].imshow((images[i * num_cols + j] * 255).astype(np.uint8))
            axes[i, j].axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Save or return (optional) image grid with filename.jpg
    if return_figure:
        fig.canvas.draw()
        out = np.array(fig.canvas.renderer._renderer)
        plt.close()
        return out
    else:
        plt.savefig(save_dir / f"{filename}.jpg")
        plt.close()


def get_wandb_image(pred_image, gt_image=None):
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.detach().cpu().numpy()
    if gt_image is not None and isinstance(gt_image, torch.Tensor):
        gt_image = gt_image.detach().cpu().numpy()
    
    pred_fig = plot_image_grid(pred_image, return_figure=True)
    if gt_image is not None:
        gt_fig = plot_image_grid(gt_image, return_figure=True)
    
    if gt_image is not None:
        fig = np.vstack([pred_fig, gt_fig])
        fig = wandb.Image(fig, caption=f"Top: Model output / Bottom: Ground Truth")
    else:
        fig = pred_fig
        fig = wandb.Image(fig, caption="Model output")
    return fig


def get_wandb_video(video, fps=30):
    """
    Video format must be `gif`, `mp4`, `webm` or `ogg`. It can be initialized
    with a numpy tensor which is either 4 (time, channel, height, width) or
    5 (batch, time, channel, height, width) dimensional.
    """
    # TODO: check if it works, additional functions might be needed.
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    
    return wandb.Video(video, fps=fps)