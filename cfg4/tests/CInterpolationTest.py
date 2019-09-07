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

::

   ipython -i  CInterpolationTest.py
   ipython -i  CInterpolationTest.py -- --nointerpol


"""
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main 
from opticks.ana.proplib import PropLib

np.set_printoptions(precision=3, suppress=True)



def check_exists(xpath, msg):
    if not os.path.exists(xpath):
        log.info(" NO SUCH PATH %s run \"%s\" " % (xpath, msg) )
        sys.exit(0)


def load(ok):
    if ok.interpol:
        path = "$TMP/InterpolationTest/CInterpolationTest_interpol.npy" 
        x_shape = (123, 4, 2, 761, 4)
        msg = "CInterpolationTest"
    else:
        path = "$TMP/InterpolationTest/CInterpolationTest_identity.npy" 
        x_shape = (123, 4, 2, 39, 4)
        msg = "CInterpolationTest --nointerpol"
    pass
    xpath = os.path.expandvars(path)
    check_exists(xpath, msg)
    c = np.load(xpath)

    log.info("load %s %r " % (xpath, c.shape))
    assert c.shape == x_shape, "shape mismatch expect %r found %r " % (x_shape, c.shape)
    return  c


class CFProp(object): 
    """
    ::

        In [23]: cf.t.shape
        Out[23]: (123, 4, 2, 39, 4)


    """
    def __init__(self, ok):
        blib = PropLib("GBndLib")
        names = blib.names
        t = np.load(os.path.expandvars("$IDPATH/GBndLib/GBndLib.npy"))
        c = load(ok)

        assert len(t) == len(names)
        assert len(t) == len(c) 
        n = len(t)

        self.shape = t.shape 
        self.names = names
        self.t = t
        self.c = c
        self.n = n

        self.consistency_check("t",t)
   


    def consistency_check(self,key,x):
        log.info(key)
        chk = {}
        for i in range(self.n):
            self.index = i 
            bnd = self.names[i]
            omat,osur,isur,imat = bnd.split("/")

            



    def __getitem__(self, sli):
        self.sli = sli 
        return self

    def __call__(self, arg):
        if type(arg) is int:
            name = self.names[arg]
        elif type(arg) is str:
            name = arg
        elif type(arg) is slice:
            return map(lambda name:self(name), self.names[arg])
        else:
            assert 0, (type(arg), "unexpected type")
        pass
        return self.check_bnd(name)

    def check_bnd(self, name):
        omat,osur,isur,imat = name.split("/")



if __name__ == '__main__':

    ok = opticks_main()

    blib = PropLib("GBndLib")
    names = blib.names
    n = len(names)

    nam = np.zeros( (n,4 ), dtype="|S64")
    for i,name in enumerate(names):nam[i] = name.split("/")




    cf = CFProp(ok)



if 0:
    for i in range(n):

        name = names[i]

        g4_omat = np.all( t[i,blib.B_OMAT,0] == c[i,blib.B_OMAT,0] )
        g4_imat = np.all( t[i,blib.B_IMAT,0] == c[i,blib.B_IMAT,0] )


        if omat in g4:
            assert g4[omat] == g4_omat
        else:
            g4[omat] = g4_omat

        if imat in g4:
            assert g4[imat] == g4_imat
        else:
            g4[imat] = g4_imat
 

        if len(osur) > 0:
            g4_osur = np.all( t[i,blib.B_OSUR,0] == c[i,blib.B_OSUR,0] )

            if osur in g4:
                assert g4[osur] == g4_osur
            else:
                g4[osur] = g4_osur
        else:
            g4_osur = None
        pass

     
        if len(isur) > 0:
            g4_isur = np.all( t[i,blib.B_ISUR,0] == c[i,blib.B_ISUR,0] )

            if isur in g4:
                assert g4[isur] == g4_isur
            else:
                g4[isur] = g4_isur
        else:
            g4_isur = None
        pass


        if g4_omat == False:
           if not omat in g4b:
               g4b[omat] = []
           g4b[omat].append( (i,blib.B_OMAT,0) )  

        if g4_osur == False:
           if not osur in g4b:
               g4b[osur] = []
           g4b[osur].append( (i,blib.B_OSUR,0) )  

        if g4_isur == False:
           if not isur in g4b:
               g4b[isur] = []
           g4b[isur].append( (i,blib.B_ISUR,0) )  

        if g4_imat == False:
           if not imat in g4b:
               g4b[imat] = []
           g4b[imat].append( (i,blib.B_IMAT,0) )  
 

        print "%4d omat %25s imat %25s         g4_omat %7s g4_imat %7s " % (  i, omat, imat,  g4_omat, g4_imat )  

        #if len(isur) > 0 or len(osur) > 0:
        #    print "%4d osur %35s isur %35s         ok_osur %7s ok_isur %7s      g4_osur %7s g4_isur %7s " % (  i, osur, isur, ok_osur, ok_isur,  g4_osur, g4_isur )  
              
 


    pass


    print "g4", g4

    for k,v in g4b.items():
        if len(v) > 0:print k, str(v)





