#! /usr/bin/python3

import shutil
import os
import glob
from astropy.io import fits
import argparse


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        help='''directory to sort files''')
    pargs = parser.parse_args(args[1:])

    dirname = pargs.dir
    files = glob.glob(dirname + '*.fits')
    objects = [fits.getheader(x)['OBJECT'] for x in files]
    objs = set(objects)
    for obj in objs:
        try:
            os.mkdir(dirname + obj)
        except FileExistsError:
            pass

    for f, o in zip(files, objects):
        shutil.move(f, dirname + o)

    return(0)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
