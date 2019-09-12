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
profilesmry.py
===============

::

    an ; ip profilesmry.py    
      ## loads times from scans, after manual adjustment of pfx and cat startswith in __main__

"""

from __future__ import print_function
import os, sys, logging, numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

from opticks.ana.num import Num
from opticks.ana.base import findfile
from opticks.ana.profile import Profile

class ProfileSmry(object):
    """
    ProfileSmry finds and loads profiles and holds values obtained 
    """
    @classmethod
    def LoadHit_(cls, htpath):
        return np.load(htpath) if os.path.exists(htpath) else None


    @classmethod
    def Load_(cls, pfx, base="$TMP", startswith=None):
        """
        :param pfx: prefix, see scan-vi, eg scan-ph-0
        :param base: directory 
        :param startswith: used to select the *cat* category of runs
                           the cat is the path element after evt, 
                           an example of *cat* for scan-ph is cvd_0_rtx_1_100M

        :return s: odict keyed by cat with Profile instances 

        Finds all persisted profiles with selected prefix that meet the startswith selection, 
        collecting them into an odict which is returned.
        """ 
        base = os.path.expandvars(os.path.join(base,pfx))

        if startswith is None:
            select_ = lambda n:True
        else:  
            select_ = lambda n:n.find(startswith) == 0   
        pass

        relp = findfile(base, Profile.NAME )   # list of relative paths beneath base
        s = odict() 
        for rp in relp:
            path = os.path.join(base, rp)
            elem = path.split("/")
            cat = elem[elem.index("evt")+1]  

            if not select_(cat): continue
            name = cat 
            ecat = cat.split("_")
            npho = Num.Int( ecat[-1] )  

            dir_ = os.path.dirname(path)

            prof = Profile(dir_, name)
            prof.npho = npho 

            htpath1 = os.path.join(dir_, "1", "ht.npy")
            ht = cls.LoadHit_(htpath1)
            nhit = ht.shape[0] if not ht is None else -1
            prof.ht = ht
            prof.nhit = nhit

            ihit = prof.npho/prof.nhit

            print("car %20s npho %9d nhit %9d ihit %5d     path %s " % (cat, prof.npho, prof.nhit, ihit,  path))
            s[cat] = prof
        pass
        return s  


    @classmethod
    def Load(cls, pfx, base="$TMP", startswith=None):
        s = cls.Load_(pfx, base, startswith)
        ps = cls.FromDict(s)
        ps.base = base
        return ps 
        
    @classmethod
    def FromDict(cls, s):
        """
        :param s: raw odict keyed with cat with Profile instance values

        Creates ProfileSmry instance comprising arrays populated
        from the Profile instances 
        """
        launch = np.zeros( [len(s), 10], dtype=np.float32 )  
        alaunch = np.zeros( [len(s) ], dtype=np.float32 )  
        interval = np.zeros( [len(s), 9], dtype=np.float32 )  
        ainterval = np.zeros( [len(s)], dtype=np.float32 )  
        npho = np.zeros( len(s), dtype=np.int32 ) 
        nhit = np.zeros( len(s), dtype=np.int32 ) 

        for i, kv in enumerate(sorted(s.items(), key=lambda kv:kv[1].npho )): 
            launch[i] = kv[1].launch
            alaunch[i] = np.average( launch[i][1:] )
            interval[i] = kv[1].start_interval
            ainterval[i] = np.average( interval[i][1:] )
            npho[i] = kv[1].npho
            nhit[i] = kv[1].nhit
        pass

        ps = cls(s)
        ps.launch = launch  
        ps.alaunch = alaunch  
        ps.interval = interval 
        ps.ainterval = ainterval 
        ps.npho = npho
        ps.nhit = nhit
        return ps 


    @classmethod
    def FromExtrapolation(cls, npho, time_for_1M=100. ):
        s = odict()
        ps = cls(s)
        ps.npho = npho
        xtim =  (npho/1e6)*time_for_1M 
        ps.alaunch = xtim
        ps.ainterval = xtim
        return ps  

    @classmethod
    def FromAB(cls, a, b, att="ainterval"):
        s = odict()
        ps = cls(s)
        assert np.all( a.npho == b.npho )
        ps.npho = a.npho
        ps.ratio = getattr(b,att)/getattr(a, att)
        return ps  

    @classmethod
    def FromAtt(cls, a, num_att="ainterval", den_att="alaunch" ):
        s = odict()
        ps = cls(s)
        ps.npho = a.npho
        ps.ratio = getattr(a,num_att)/getattr(a, den_att)
        return ps  


    def __init__(self, s):
        self.s = s 

    def __repr__(self):
        return "\n".join(["ProfileSmry", Profile.Labels()]+map(lambda kv:repr(kv[1]), sorted(self.s.items(), key=lambda kv:kv[1].npho )  ))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=200)

    ps = {}

    pfx = "scan-ph-0"

    ps[0] = ProfileSmry.Load(pfx, startswith="cvd_0_rtx_0")
    ps[1] = ProfileSmry.Load(pfx, startswith="cvd_0_rtx_1")
    #ps[10] = ProfileSmry.Load("scan-ph-tri", startswith="cvd_1_rtx_1")
    ps[9] = ProfileSmry.FromExtrapolation( ps[0].npho,  time_for_1M=100. )


