Utility codes for Deep Learning(logger, base modules, metrics, visualizers, etc.).
My version of [kitsu](https://github.com/Kitsunetic/kitsu), thanks to J.H. Shim !!

# Contents

```bash
.
|-- __init__.py
|-- metrics
|   |-- __init__.py
|   |-- evaluation_metrics.py
|   |-- pointcloud.py
|   `-- score # Inception and FID score calculation
|       |-- __init__.py
|       |-- fid.py
|       `-- inception.py
|-- modules
|   |-- __init__.py # Including common functions for modules
|   |-- diffusion # Diffusion Trainer(DDPM), and Samplers(DDPM, DDIM, Karras)
|   |   |-- __init__.py # Including common functions for Diffusion (e.g. get_betas())
|   |   |-- ddim.py
|   |   |-- ddpm.py
|   |   |-- diffusion_base.py
|   |   `-- karras.py
|   `-- unet # Unet models for Diffusion models (for 1D and 2D)
|       |-- __init__.py
|       |-- base_modules.py
|       |-- ldm_unet.py # Unet model for Latent Diffusion Model
|       |-- transformer.py
|       |-- unet_base.py # Base Unet models for general use
|       `-- unet_modules.py
|-- pointcloud
|   |-- __init__.py
|   `-- transform.py
`-- utils
    |-- __init__.py
    |-- data.py
    |-- dist.py
    |-- ema.py
    |-- indexing.py
    |-- interp1d.py
    |-- io.py
    |-- logger.py # Customized logger
    |-- optim.py
    |-- resize_right.py # from https://github.com/assafshocher/ResizeRight
    |-- sched.py # Useful learning rate schedulers
    |-- utils.py
    `-- vis3d.py # Visualization of 3D Scenes
```


# To-do List

- [ ] Test 1D U-Net
- [ ] Implement Conditional U-Net (need to study classifier free guidance)
- [ ] Trouble shooting of LDM U-Net (OOM)
- [ ] Implement 3D U-Net
- [ ] Organize metrics, pointcloud, utils