import os

import pandas as pd

def get_data(path):
    '''
    Imports data formatted according to speech_augment
    
    :param path: root directory of dataset (contains clips dir, train.csv, test.csv and val.csv)

    :return : train df, test df, validation df
    '''

    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    val_df = pd.read_csv(os.path.join(path, 'val.csv'))

    return train_df, test_df, val_df