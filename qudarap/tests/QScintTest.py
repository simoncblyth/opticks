#!/usr/bin/env python
"""

~/o/qudarap/tests/QScintTest.sh pdb
~/o/qudarap/tests/QScintTest.sh ana


"""
import sys, os
import numpy as np
from opticks.ana.fold import Fold


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    src = f.src
    dst = f.dst

    src_dst_equal = np.all( src == dst )
    src_dst_close = np.allclose( src, dst )
    rc = 0 if src_dst_equal else 1
    print(f" src_dst_equal {src_dst_equal} src_dst_close {src_dst_close} rc {rc} ")

    sys.exit(rc)
