#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""

This script is invoked from oks-dbgseed, which 
first creates the seeds with commands like::

   GGeoViewTest --dbgseed --trivial --cerenkov --compute
   GGeoViewTest --dbgseed --trivial --cerenkov 

Looks like on Linux the interop photon buffer is being seeded but is 
then getting overwritten ??

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
           ...
           [-918184732,         18],
           [-918183179,         21],
           [-918181729,        108],
           [-918180539,       7062],
           [-918178532,         12],
           [-918178359,     368294],
           [         0,         80]])


Things go wrong from item 11883::

    In [41]: i[11880:12000,0,0].view(np.int32)
    Out[41]: 
    array([       160,        160,        160, -965867012, -965867012,
           -965867012, -965867012, -965867012, -965852569, -965852569,
           -965852569, -965852569, -965852569, -965852569, -965852569,
           -965852569, -965852569, -965852569, -965852569, -965852569,
           -965852569, -965852569, -965852569, -965852569, -965852569,
           -965852569, -965852569, -965852569, -965852569, -965852569,
           -965852569, -965852569, -965852569, -965852569, -965852569,
           -965852569, -965852569, -965852569, -965852569, -965852569,


    In [42]: i.shape
    Out[42]: (612841, 4, 4)

::

    In [49]: i[11882]
    Out[49]: 
    array([[      0.   ,       0.   ,       0.   ,       0.   ],
           [-806767.188,   -8289.5  ,       0.   ,       0.   ],
           [ -15244.033, -806768.125,   -8315.324,  -15232.615],
           [-806767.188,   -8289.5  ,       0.   ,       0.   ]], dtype=float32)

    In [50]: i[11883]
    Out[50]: 
    array([[ -15234.496, -806782.125,   -8417.453,  -15215.982],
           [-806768.062,   -8395.   ,         nan,       0.   ],
           [ -15234.496, -806782.125,   -8290.   ,  -15215.982],
           [-806768.062,   -8267.547,         nan,       0.   ]], dtype=float32)


    In [47]: i[11882].view(np.int32)
    Out[47]: 
    array([[       160,          0,          0,          0],
           [-918227213, -972978688,     227185,     227189],
           [-965857246, -918227198, -972952244, -965868938],
           [-918227213, -972978688,     227189,     227191]], dtype=int32)

    In [48]: i[11883].view(np.int32)
    Out[48]: 
    array([[-965867012, -918226974, -972847664, -965885970],
           [-918227199, -972870656,         -1,     317936],
           [-965867012, -918226974, -972978176, -965885970],
           [-918227199, -973001168,         -1,     317944]], dtype=int32)




"""

import os, sys, datetime, logging, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.nbase import count_unique

log = logging.getLogger(__name__)


def stmp_(st, fmt="%Y%m%d-%H%M"): 
    return datetime.datetime.fromtimestamp(st.st_ctime).strftime(fmt)

def stamp_(path, fmt="%Y%m%d-%H%M"):
    try:
        st = os.stat(path)
    except OSError:
        return "FILE-DOES-NOT-EXIST"
    return stmp_(st, fmt=fmt)

def x_(_):
    p = os.path.expandvars(_)
    st = stamp_(p)
    log.info( " %s -> %s (%s) " % (_, p, st))
    return p  

def check_dbgseed(a,g):
    """
    The seeds should be genstep_id from 0:num_genstep-1 
    """
    aa = count_unique(a[:,0,0].view(np.int32))
    assert np.all(aa[:,0] == np.arange(0,len(aa)))

    xx = g[:,0,3].view(np.int32)   ## photons per genstep

    assert len(aa) == len(xx)
    assert np.all(aa[:,1] == xx)


if __name__ == '__main__':
    args = opticks_main(src="torch", tag="1", det="dayabay")

    np.set_printoptions(suppress=True, precision=3)

    cpath = x_("$TMP/dbgseed_compute.npy")
    ipath = x_("$TMP/dbgseed_interop.npy")

    log.info("cpath : %s " % cpath) 
    log.info("ipath : %s " % ipath) 

    if not(os.path.exists(cpath) and os.path.exists(ipath)):
        log.warning("SKIP due to missing path")
        sys.exit(0)  ## very particular test, not a standardized test yet so dont treat as a fail 

    c = np.load(cpath)
    i = np.load(ipath)

    log.info(" c : %s " % repr(c.shape) )
    log.info(" i : %s " % repr(i.shape) )

    cj = c[:,0,0].view(np.int32)
    ij = i[:,0,0].view(np.int32)


    if args.src in ("cerenkov", "scintillation"):
        g = np.load(x_("$OPTICKS_DATA_DIR/gensteps/%s/%s/%s.npy" % (args.det,args.src,args.tag) ))
    elif args.src == "torch":
        g = np.load(x_("$TMP/torchdbg.npy"))
    else:
        assert 0, args.src

    if args.src == "torch":
        print "cj", cj[99500:]
        print "ij", ij[99500:]
    pass

    check_dbgseed(c,g)
    check_dbgseed(i,g)

    ii = count_unique(i[:,0,0].view(np.int32))
    cc = count_unique(c[:,0,0].view(np.int32))

  

    



