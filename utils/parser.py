from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, help='location of config file', default='config/config.yaml')

    return parser.parse_args()