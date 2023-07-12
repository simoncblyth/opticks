#!/usr/bin/env python

"""

Note the repetition wraparound::

    t.prd.view(np.int32)[:,:,1,:]
    [[[  0   0 100  20]
      [  0   0 200  19]
      [  0   0 300  29]
      [  0   0 400  39]
      [  0   0 100  20]
      [  0   0 200  19]]

     [[  0   0 100  20]
      [  0   0 200  19]
      [  0   0 300  29]
      [  0   0 400  39]
      [  0   0 100  20]
      [  0   0 200  19]]

"""

import numpy as np, textwrap
from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    EXPR = list(filter(None,textwrap.dedent(r"""
    t.prd.view(np.int32)[:,:,1,:]
    """).split("\n")))

    for expr in EXPR:
        print(expr)
        print(eval(expr))
    pass

