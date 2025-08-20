'''
Script to compress exising pickle file.
'''

import argparse
from pathlib import Path
from optilab.utils.pickle_utils import load_from_pickle, dump_to_pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=Path, help='Path to input pickle.',)
    parser.add_argument('output_path', type=Path, help='Path to save the pickle to.')
    parser.add_argument('compression', type=int, help='Zstd compression level.', default=1)
    args = parser.parse_args()

    print('starting')
    data = load_from_pickle(args.input_path)
    print('pickle read')
    dump_to_pickle(data, args.output_path, zstd_compression=args.compression)
    print('pickle dumped')