#!/usr/bin/env python
import numpy as np
from opticks.ana.fold import Fold


"""
+----+-----------------------+----------------+-----------------------+----------------+------------------------------+
| q  |      x                |      y         |     z                 |      w         |  notes                       |
+====+=======================+================+=======================+================+==============================+
|    | u:hitcount_identity   |  f:time        | u:lposcost_lposfphi   | u:flagmask     |                              |
| q0 |                       |                |                       |                |                              |
|    | off:0, 2              | off:4          | off:8,10              | off:12         |  off:byte offsets            |
+----+-----------------------+----------------+-----------------------+----------------+------------------------------+
"""


dtype = np.dtype({
    'names':   ['identity', 'hitcount', 'time',     'lposfphi', 'lposcost', 'flagmask'],
    'formats': [np.uint16,  np.uint16,  np.float32, np.uint16,   np.uint16,   np.uint32],
    'offsets': [0,          2,          4,          8,          10,          12],
    'itemsize': 16
})

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    p = t.demoarray.view(dtype)

pass


