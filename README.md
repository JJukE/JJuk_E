Own framework and utilities for Deep Learning(trainer, logger, metrics, visualizers, useful utilities, etc.).
My version of [kitsu](https://github.com/Kitsunetic/kitsu), thanks to J.H. Shim !!

# Contents

```bash
.
|-- README.md
|-- __init__.py
|-- datasets
|   `-- dataloaders.py
|-- main.py
|-- metrics
|   |-- __init__.py
|   |-- __pycache__
|   |   |-- __init__.cpython-38.pyc
|   |   |-- evaluation_metrics.cpython-38.pyc
|   |   `-- pointcloud.cpython-38.pyc
|   |-- evaluation_metrics.py
|   |-- pointcloud.py
|   `-- score
|       |-- __init__.py
|       |-- __pycache__
|       |   |-- __init__.cpython-38.pyc
|       |   |-- fid.cpython-38.pyc
|       |   `-- inception.cpython-38.pyc
|       |-- fid.py
|       `-- inception.py
|-- models
|   |-- ema_trainer.py
|   |-- optimizer.py
|   |-- scheduler.py
|   `-- trainer.py
|-- net_utils
|   |-- __init__.py
|   |-- __pycache__
|   |   |-- __init__.cpython-38.pyc
|   |   |-- dist.cpython-38.pyc
|   |   |-- logger.cpython-38.pyc
|   |   |-- options.cpython-38.pyc
|   |   `-- utils.cpython-38.pyc
|   |-- dist.py
|   |-- logger.py
|   |-- options.py
|   `-- utils.py
`-- utils
    |-- __init__.py
    |-- __pycache__
    |   |-- __init__.cpython-38.pyc
    |   |-- data.cpython-38.pyc
    |   |-- dist.cpython-38.pyc
    |   |-- ema.cpython-38.pyc
    |   |-- indexing.cpython-38.pyc
    |   |-- interp1d.cpython-38.pyc
    |   |-- io.cpython-38.pyc
    |   |-- logger.cpython-38.pyc
    |   |-- optim.cpython-38.pyc
    |   |-- resize_right.cpython-38.pyc
    |   |-- sched.cpython-38.pyc
    |   |-- utils.cpython-38.pyc
    |   `-- vis3d.cpython-38.pyc
    |-- interp1d.py
    |-- resize_right.py
    |-- utils(to_be_fixed).py
    `-- vis3d.py
```

# To-do List

- [ ] Test framework