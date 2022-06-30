#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
    path = "/tmp/logTest.npy"
    print(path)
    a = np.load(path)
    print(a)

