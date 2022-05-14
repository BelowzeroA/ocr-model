import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import os
import argparse

from common.file_utils import Utils

utils = Utils()


def convert():
    data_dir = utils.path_from_root('data')
    lines = utils.load_list_from_file(os.path.join(data_dir, 'Export.txt'), encoding='windows-1251')
    filenames, texts = [], []
    for line in lines:
        parts = line.split()
        filename = parts[0]
        text = parts[2]
        filenames.append(f'pics/{filename}.jpg')
        texts.append(text.strip())

    data = {'filename': filenames, 'text': texts}
    df = pd.DataFrame(data)
    pivot = np.random.rand(len(df)) < 0.9
    train = df[pivot]
    test = df[~pivot]
    # train = train.sample(n=100)
    # test = test.sample(n=100)
    train.to_csv(os.path.join(data_dir, 'train.csv'))
    test.to_csv(os.path.join(data_dir, 'val.csv'))


if __name__ == '__main__':
    convert()
