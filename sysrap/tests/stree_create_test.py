#!/usr/bin/env python

import numpy as np
from numpy.linalg import multi_dot

from opticks.ana.fold import Fold
from opticks.ana.eprint import eprint, epr
from opticks.sysrap.stree import stree, snode, snd

if __name__ == '__main__':

    snode.Type()
    snd.Type()

    f = Fold.Load("$FOLD/stree", symbol="f")
    print(repr(f))

    st = stree(f)
    print(repr(st))



