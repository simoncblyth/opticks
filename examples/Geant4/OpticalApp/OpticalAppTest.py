#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
    FOLD = os.environ["FOLD"]
    r = np.load(os.path.join(FOLD, "record.npy"))
    print(repr(r))
    q = np.load(os.path.join(FOLD, "seq.npy"))
    print(repr(q))
pass
