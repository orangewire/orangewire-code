#!/usr/bin/python3

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import sys, argparse


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to data file')
    parser.add_argument('-t', '--title', default='[matrix]', help='Plot title')
    args = parser.parse_args()

    data_file = args.file
    if not Path(data_file).is_file():
        print('File does not exist')
        return

    data = np.loadtxt(data_file, dtype=int)
    plt.imshow(data, interpolation='nearest')
    plt.gray()
    plt.suptitle(args.title)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
