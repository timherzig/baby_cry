from omegaconf import OmegaConf

from data.get_data import get_data

from utils.parser import parse_arguments

def main(args):
    config = OmegaConf.load(args.config)

    train_df, test_df, val_df = get_data(config.data.path)

    model = 

    print('Done')

if __name__ == '__main__':
    args = parse_arguments()

    main(args)