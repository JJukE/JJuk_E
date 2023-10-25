Own framework and utilities for Deep Learning(trainer, logger, metrics, visualizers, useful utilities, etc.).
My version of [kitsu](https://github.com/Kitsunetic/kitsu), thanks to J.H. Shim !!

# Contents

```bash
.
|-- README.md
|-- __init__.py
|-- datasets
|   |-- __init__.py
|   `-- dataloaders.py
|-- main.py
|-- metrics
|   |-- __init__.py
|   |-- evaluation_metrics.py
|   |-- pointcloud.py
|   `-- score
|       |-- __init__.py
|       |-- fid.py
|       `-- inception.py
|-- models
|   |-- __init__.py
|   |-- ema_trainer.py
|   |-- optimizer.py
|   |-- scheduler.py
|   `-- trainer.py
|-- net_utils
|   |-- __init__.py
|   |-- dist.py
|   |-- logger.py
|   |-- options.py
|   `-- utils.py
`-- utils
    |-- __init__.py
    |-- interp1d.py
    |-- resize_right.py
    |-- utils.py
    `-- vis3d.py
```

# To-do List

- [ ] Test framework