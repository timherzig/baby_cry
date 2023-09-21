from datetime import datetime
from lightning.pytorch.loggers import WandbLogger


def prepare(config, args):
    logger = None
    if config.logger == "wandb":
        name = f'{config.mode}_{config.model.name}_{config.data.dataset}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}'

        logger = WandbLogger(**config.wandb, name=name, save_dir=f"checkpoints/{name}")
        logger.log_hyperparams(config.train_args)

    return logger
