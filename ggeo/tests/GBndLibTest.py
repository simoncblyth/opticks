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
Compare old GBoundaryLib buffer with new GBndLib dynamically created oned

To get indices agreement in the optical buffer had to::

  cp GSurfaceIndexLocal.json   ~/.opticks/GSurfaceLib/order.json 
  cp GMaterialIndexLocal.json  ~/.opticks/GMaterialLib/order.json 

  ggv -G       # recreate geocache
  ggv --bnd    # recreate bnd buffers
  ggv --pybnd  # run this test

Why is order changing on each run ?  Probably not, presumably 
this was due to the move from digest based identity to shortname based identity. 

"""
import os
import numpy as np
from opticks.ana.base import opticks_main


path_ = lambda _:os.path.expandvars("$IDPATH/%s" % _)
load_ = lambda _:np.load(path_(_))

def test_old_boundary_buffer():
    w = load_("wavelength.npy").reshape(-1,4,39,4)
    b = load_("GBndLib/GBndLib.npy")
    assert w.shape == b.shape , ( w.shape, b.shape )
    num_quad = 4 
    for j in range(num_quad):
        print "quad %d " % j 
        for i in range(len(w)):
            assert np.all( w[i][j] == b[i][j] ) 
        pass
    assert np.all( w == b ) == True 


def test_old_optical_buffer():
    # huh indices in optical.npy 1-based ? Yep npy- Index::add sets local indices like this
    w = load_("optical.npy").reshape(-1,4,4)  
    b = load_("GBndLib/GBndLibOptical.npy")

    assert w.shape == b.shape , ( w.shape, b.shape )
    for i in range(len(w)):
        if not np.all( w[i] == b[i] ):
            print 
            print i
            print w[i]
            print b[i] 
    pass
    assert np.all( w == b ) == True 


def test_buffers():
    names = ["GBndLib/GBndLib.npy","GBndLib/GBndLibIndex.npy","GBndLib/GBndLibOptical.npy"]
    for name in names:
        path = path_(name)
        if os.path.exists(path):
            os.system("ls -l %s " % path)
    for name in names:
        path = path_(name)
        if os.path.exists(path):
            buf = np.load(path)
            print "%40s %s " % (name, repr(buf.shape))
            print buf 


if __name__ == '__main__':

    args = opticks_main()
    print args 
    test_buffers()


