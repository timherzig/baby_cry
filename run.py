from omegaconf import OmegaConf

from src.utils.parser import parse_arguments
from src.train import train
from src.test import test

from src.data.data import DataModule
from src.model.model import Model

from src.utils.prepare import prepare


def main(args):
    print(f"Starting")
    print(f"-------------------")
    config = OmegaConf.load(args.config)
    print(f"Config: {config} loadedl")
    print(f"-------------------")

    logger = prepare(config, args)
    print(f"Stats:")
    print(f"  logger: {logger}")
    print(f"  n workers: {args.num_workers}")
    print(f"-------------------")

    datamodule = DataModule(config, num_workers=args.num_workers)
    print(f"DataModule: {datamodule} created")
    print(f"-------------------")

    model = Model(config)

    print(f"Running {config.mode} mode")
    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        print("Invalid mode")
        raise NotImplementedError
    print(f"-------------------")

    print("Done.")


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
