#!/usr/bin/env python
"""
Looks like interop photon buffer is seeded correctly but is then getting
overwritten ??

::

    In [22]: cc[:10]
    Out[22]: 
    array([[  0,  80],
           [  1, 108],
           [  2,  77],
           [  3,  30],
           [  4,  99],
           [  5, 105],
           [  6, 106],
           [  7,  85],
           [  8,  94],
           [  9,  29]])

    In [23]: ii[42:42+10]
    Out[23]: 
    array([[  0,  80],
           [  1, 108],
           [  2,  77],
           [  3,  30],
           [  4,  99],
           [  5, 105],
           [  6, 106],
           [  7,  85],
           [  8,  94],
           [  9,  29]])


    n [11]: ii[:43]
    Out[11]: 
    array([[-965867012,          5],
           [-965852569,     192504],
           [-961979442,          4],
           [-961978135,         54],
           [-961921942,         49],
           [-961761495,        286],
           [-918320253,          6],
           [-918316397,          3],
           [-918313480,         12],
           [-918304859,        810],
           [-918304088,          9],
           [-918303887,       5751],
           [-918303335,        105],
           [-918303034,       5277],
           [-918253936,          6],
           [-918230512,         15],
           [-918224235,      13230],
           [-918214216,         18],
           [-918213650,         18],
           [-918213547,          9],
           [-918210656,        369],
           [-918209659,        543],
           [-918207048,          3],
           [-918206296,         15],
           [-918205230,          3],
           [-918204769,         21],
           [-918202251,         12],
           [-918202109,          3],
           [-918201802,         15],
           [-918200388,          3],
           [-918200249,         21],
           [-918197609,       6210],
           [-918192256,         18],
           [-918190632,         21],
           [-918188956,          6],
           [-918188198,          9],
           [-918184732,         18],
           [-918183179,         21],
           [-918181729,        108],
           [-918180539,       7062],
           [-918178532,         12],
           [-918178359,     368294],
           [         0,         80]])



"""

import os, sys, logging, numpy as np
from opticks.ana.base import opticks_environment
from opticks.ana.base import opticks_args
from opticks.ana.nbase import count_unique

log = logging.getLogger(__name__)
x_ = lambda _:os.path.expandvars(_)


def check_dbgseed():
    """
    Create these arrays with::

        GGeoViewTest --dbgseed --trivial --compute --cerenkov 
        GGeoViewTest --dbgseed --trivial --cerenkov 

    """
    i = np.load(x_("$TMP/dbgseed_interop.npy"))
    c = np.load(x_("$TMP/dbgseed_compute.npy"))

    check_dbgseed_array(c)
    check_dbgseed_array(i)

    return i,c 


def check_dbgseed_array(a, gspath="$OPTICKS_DATA_DIR/gensteps/dayabay/cerenkov/1.npy"):
    """
    The seeds should be genstep_id from 0:num_genstep-1 
    """
    aa = count_unique(a[:,0,0].view(np.int32))
    assert np.all(aa[:,0] == np.arange(0,len(aa)))

    gs = np.load(x_(gspath))
    xx = gs[:,0,3].view(np.int32)   ## photons per genstep

    assert len(aa) == len(xx)
    assert np.all(aa[:,1] == xx)


if __name__ == '__main__':
    opticks_environment()
    args = opticks_args()

    i,c = check_dbgseed()

    ii = count_unique(i[:,0,0].view(np.int32))
    cc = count_unique(c[:,0,0].view(np.int32))


  

    



