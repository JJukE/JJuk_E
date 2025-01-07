from jjuke import logger, options


def train():
    args = options.get_config()
    
    logger.basic_config(args.exp_path / "train.log")
    args.log = logger.get_logger()
    
    trainer = options.instantiate_from_config(args.trainer, args)
    trainer.fit()


if __name__ == "__main__":
    train()
