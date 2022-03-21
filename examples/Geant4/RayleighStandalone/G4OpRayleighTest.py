#!/usr/bin/env python

import os, numpy as np

FOLD = "/tmp/G4OpRayleighTest"

if __name__ == '__main__':
     p = np.load(os.path.join(FOLD, "p.npy"))
     print(p.shape)


