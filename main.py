from omegaconf import OmegaConf
from src.utils.parser import parse_args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    if args.task == "train":
        from src.train import train

        train(config, args.config.split("/")[-1].split(".")[0])

    elif args.task == "test":
        from src.test import test

        test(config)

    elif args.task == "pitch_distribution":
        from src.stats.pitch_distribution import pitch_distribution

        pitch_distribution(config)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    print("Done!")
