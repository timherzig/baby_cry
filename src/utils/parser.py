from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        help="task to run",
        default="train",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="location of config file",
        default="config/config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint to load",
        default=None,
    )

    return parser.parse_args()
