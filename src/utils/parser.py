from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="location of config file",
        default="config/config.yaml",
    )
    parser.add_argument("--num_workers", type=int, help="number of workers", default=4)

    return parser.parse_args()
