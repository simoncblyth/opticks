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

    x_hitcount = np.arange(0, 10, dtype=np.uint32)
    x_identity = np.arange(0,10000,1000, dtype=np.uint32)
    x_time = np.arange(0, 1, 0.1, dtype=np.float32)
    x_lposcost = np.repeat(0.5, 10).astype(np.float32)
    x_lposfphi = np.repeat(0.6, 10).astype(np.float32)
    x_flagmask = np.repeat( 0x2040, 10 ).astype(np.uint32)


    assert( np.all( x_hitcount == p['hitcount'].ravel() ))
    assert( np.all( x_hitcount == t.demoarray.view(np.uint32)[:,0] >> 16  ))

    assert( np.all( x_identity == p['identity'].ravel() ))
    assert( np.all( x_identity == t.demoarray.view(np.uint32)[:,0] & 0xFFFF  ))

    assert( np.all( x_time == p['time'].ravel() ))
    assert( np.all( x_time == t.demoarray[:,1] ))



    epsilon = 1e-5

    assert( np.all( np.abs( x_lposcost - p['lposcost'].ravel().astype(np.float32)/0xffff ) < epsilon  )   )
    assert( np.all( np.abs( x_lposcost - ((t.demoarray[:,2].view(np.uint32) >> 16)/0xffff).astype(np.float32) < epsilon )) )

    assert( np.all( np.abs( x_lposfphi - p['lposfphi'].ravel().astype(np.float32)/0xffff )  < epsilon ) )
    assert( np.all( np.abs( x_lposfphi - ((t.demoarray[:,2].view(np.uint32) & 0xffff)/0xffff).astype(np.float32) < epsilon )) )

    assert( np.all( x_flagmask == p['flagmask'].ravel() ))
    assert( np.all( x_flagmask == t.demoarray[:,3].view(np.uint32) ))


pass


