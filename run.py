from lightning import Trainer
from omegaconf import OmegaConf

from data.data_module import AugmentedDataModule
from utils.parser import parse_arguments
from src.model import Whisper_Encoder

def main(args):
    config = OmegaConf.load(args.config)

    model = Whisper_Encoder(config=config)
    trainer = Trainer()

    df = AugmentedDataModule(config.data.root)
    trainer.fit(model, datamodule=df)

    print('Done')

if __name__ == '__main__':
    args = parse_arguments()

    main(args)