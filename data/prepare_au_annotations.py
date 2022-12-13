import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-ia', '--input_aus_filesdir', type=str, default='C:/Datasets/CACD/CACD_128_align_AUS', help='dir for input file')
parser.add_argument('-op', '--output_path', type=str, default='C:/Datasets/CACD',help='out put dir')
args = parser.parse_args()

def get_data(filepaths):
    data = dict()
    for filepath in tqdm(filepaths):
        cond=np.zeros(23)
        content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
        if len(content.shape)!=1:
            cond[:17]=content[8:25]
            # cond[17:20]=content[5:8]
            name=os.path.basename(filepath[:-4])
            age=int(name.split('_')[0])
            group=0
            if age<=30:
                group=0
            elif age>=31 and age<=40:
                group=1
            elif age>=41 and age<=50:
                group=2
            else:
                group=3
            index=group+17
            cond[index]=1
            data[name] = cond

    return data


def save_dict(data, name):
    with open(name , 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def main():
    filepaths = glob.glob(os.path.join(args.input_aus_filesdir, '*.csv'))
    filepaths.sort()

    # create aus file
    data = get_data(filepaths)

    # if not os.path.isdir(args.output_path):
    #     os.makedirs(args.output_path)
    # save_dict(data, os.path.join(args.output_path, "CACD_aus_attr_whole.pkl"))


if __name__ == '__main__':
    main()
