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


import os, numpy as np
import logging
log = logging.getLogger(__name__)

from opticks.ana.proplib import PropLib

np.set_printoptions(precision=3, suppress=True)


if __name__ == '__main__':

    base = "$TMP/interpolationTest"
    blib = Proplib.load_GBndLib(base)
    names = blib.names
    t = blib.data # boundary texture data

    #interpol = False 
    interpol = True

    if interpol:
       ext, nl = "interpol", 820-60+1
    else:
       ext, nl = "identity", 39

    #oname = "OInterpolationTest_%s.npy" % ext 
    #cname = "CInterpolationTest_%s.npy" % ext 

    oname = "interpolationTest_%s.npy" % ext 
    cname = "CInterpolationTest_%s.npy" % ext 

    o = np.load(os.path.expandvars(os.path.join(base,oname))).reshape(-1,4,2,nl,4) 
    c = np.load(os.path.expandvars(os.path.join(base,cname))).reshape(-1,4,2,nl,4) 

    if not interpol:
       assert np.all(t == o)
    

    assert len(t) == len(names)
    assert len(t) == len(o) 
    assert len(t) == len(c) 
    n = len(t)

    ok={}
    g4={}
    g4b={}

    for i in range(n):

        name = names[i]
        omat,osur,isur,imat = name.split("/")

        ok_omat = np.all( t[i,blib.B_OMAT,0] == o[i,blib.B_OMAT,0] )
        ok_imat = np.all( t[i,blib.B_IMAT,0] == o[i,blib.B_IMAT,0] )

        g4_omat = np.all( t[i,blib.B_OMAT,0] == c[i,blib.B_OMAT,0] )
        g4_imat = np.all( t[i,blib.B_IMAT,0] == c[i,blib.B_IMAT,0] )


        if omat in ok:
            assert ok[omat] == ok_omat
        else:
            ok[omat] = ok_omat
 
        if imat in ok:
            assert ok[imat] == ok_imat
        else:
            ok[imat] = ok_imat
 

        if omat in g4:
            assert g4[omat] == g4_omat
        else:
            g4[omat] = g4_omat

        if g4_omat == False:
           if not omat in g4b:
               g4b[omat] = []
           g4b[omat].append( (i,blib.B_OMAT,0) )  
 
        if imat in g4:
            assert g4[imat] == g4_imat
        else:
            g4[imat] = g4_imat
 
        if g4_imat == False:
           if not imat in g4b:
               g4b[imat] = []
           g4b[imat].append( (i,blib.B_IMAT,0) )  
 

        print "%4d omat %25s imat %25s         ok_omat %7s ok_imat %7s      g4_omat %7s g4_imat %7s " % (  i, omat, imat, ok_omat, ok_imat,  g4_omat, g4_imat )  
    pass


    print "ok", ok
    print "g4", g4

    for k,v in g4b.items():
        if len(v) > 0:print k, str(v)





