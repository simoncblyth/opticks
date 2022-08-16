#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    print("repr(f.inst)")
    print(repr(f.inst))

    print("repr(f.inst.view(np.int64))")
    print(repr(f.inst.view(np.int64)))

    print("repr(f.inst_f4)")
    print(repr(f.inst_f4))

    print("repr(f.inst_f4.view(np.int32))")
    print(repr(f.inst_f4.view(np.int32)))


