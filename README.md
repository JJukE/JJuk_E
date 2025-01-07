My own framework for training Deep Learning networks, getting advantages of both [accelerate](https://huggingface.co/docs/accelerate/index) by huggingface and [kitsu](https://github.com/Kitsunetic/kitsu), which is also a customized training framework based on the pytorch-lightning by [kitsunetic](https://github.com/kitsunetic).

Used example is in [this link](https://github.com/JJukE/DanceGen)

It is optimized and tested in
```bash
conda install -c pytorch -c nvidia pytorch==2.3.1 pytorch-cuda=12.1
conda install -c conda-forge accelerate==1.0.1 transformers==4.46.1 diffusers==0.30.3
```

# Contents

Main classes to override are **BaseTrainer** in ```model/trainer.py``` for training and **instantiate_from_config** in ```util/options.py``` for configuration.

Below is the least setting for training a model. Contents ```main.py```, ```train.py```, ```config/``` and ```dataset/``` are just examples.

```bash
.
|-- config
|   `-- model_to_experiment.yaml
|-- dataset
|   |-- __init__.py
|   `-- dataloader.py
|-- model
|   |-- __init__.py
|   `-- trainer.py
|-- util
|   |-- __init__.py
|   |-- logger.py
|   `-- options.py
|-- __init__.py
`-- main.py
`-- train.py
```
