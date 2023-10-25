import types

import numpy as np
import torch
from tqdm import tqdm, trange

from .fid import calculate_frechet_distance, torch_cov
from .inception import InceptionV3

device = torch.device("cuda")

def get_inception_score(images, splits=10, batch_size=32, use_torch=False, verbose=False, parallel=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM["prob"]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    preds = []
    iterator = trange(
        0, len(images), batch_size, dynamic_ncols=True, leave=False, disable=not verbose, desc="get_inception_score"
    )

    for start in iterator:
        end = start + batch_size
        batch_images = images[start:end]
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)[0]
        if use_torch:
            preds.append(pred)
        else:
            preds.append(pred.cpu().numpy())
    if use_torch:
        preds = torch.cat(preds, 0)
    else:
        preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits) : ((i + 1) * preds.shape[0] // splits), :]
        if use_torch:
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        is_mean, is_std = (torch.mean(scores).cpu().item(), torch.std(scores).cpu().item())
    else:
        is_mean, is_std = np.mean(scores), np.std(scores)
    del preds, scores, model
    return is_mean, is_std


def get_inception_and_fid_score(
    images, fid_cache, num_images=None, splits=10, batch_size=50, use_torch=False, verbose=False, parallel=False
):
    """when `images` is a python generator, `num_images` should be given"""

    if num_images is None and isinstance(images, types.GeneratorType):
        raise ValueError("when `images` is a python generator, " "`num_images` should be given")

    if num_images is None:
        num_images = len(images)

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM["prob"]
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    if use_torch:
        fid_acts = torch.empty((num_images, 2048)).to(device)
        is_probs = torch.empty((num_images, 1008)).to(device)
    else:
        fid_acts = np.empty((num_images, 2048))
        is_probs = np.empty((num_images, 1008))

    iterator = iter(
        tqdm(
            images,
            total=num_images,
            dynamic_ncols=True,
            leave=False,
            disable=not verbose,
            desc="get_inception_and_fid_score",
            ncols=100,
        )
    )

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            if use_torch:
                fid_acts[start:end] = pred[0].view(-1, 2048)
                is_probs[start:end] = pred[1]
            else:
                fid_acts[start:end] = pred[0].view(-1, 2048).cpu().numpy()
                is_probs[start:end] = pred[1].cpu().numpy()
        start = end

    # Inception Score
    scores = []
    for i in range(splits):
        part = is_probs[(i * is_probs.shape[0] // splits) : ((i + 1) * is_probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        is_score = (torch.mean(scores).cpu().item(), torch.std(scores).cpu().item())
    else:
        is_score = (np.mean(scores), np.std(scores))

    # FID Score
    f = np.load(fid_cache)
    m2, s2 = f["mu"][:], f["sigma"][:]
    f.close()
    if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        s1 = torch_cov(fid_acts, rowvar=False)
        m2 = torch.tensor(m2).to(m1.dtype).to(device)
        s2 = torch.tensor(s2).to(s1.dtype).to(device)
    else:
        m1 = np.mean(fid_acts, axis=0)
        s1 = np.cov(fid_acts, rowvar=False)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2, use_torch=use_torch)

    del fid_acts, is_probs, scores, model
    return is_score, fid_score