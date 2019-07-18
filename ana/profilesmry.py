#!/usr/bin/env python
"""
profilesmry.py
===============

::

    ip profilesmry.py


"""

from __future__ import print_function
import os, sys, logging, numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

from opticks.ana.num import Num
from opticks.ana.base import findfile
from opticks.ana.profile import Profile

class ProfileSmry(object):
    @classmethod
    def Load(cls, pfx, base="$TMP", startswith=None):

        base = os.path.expandvars(os.path.join(base,pfx))

        if startswith is None:
            select_ = lambda n:True
        else:  
            select_ = lambda n:n.find(startswith) == 0   
        pass

        relp = findfile(base, Profile.NAME )
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
            if os.path.exists(htpath1):
                ht = np.load(htpath1)
                nhit = ht.shape[0]
                prof.ht = ht
                prof.nhit = nhit
            else:
                prof.ht = None
                prof.nhit = -1  
            pass 

            ihit = npho/nhit

            print("car %20s npho %9d nhit %9d ihit %5d     path %s " % (cat, prof.npho, prof.nhit, ihit,  path))
            s[cat] = prof
        pass

        ps = cls.FromDict(s)
        ps.base = base
        return ps 
        
    @classmethod
    def FromDict(cls, s):
 
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

    ps[0] = ProfileSmry.Load("scan-ph", startswith="cvd_1_rtx_0")
    ps[1] = ProfileSmry.Load("scan-ph", startswith="cvd_1_rtx_1")
    ps[10] = ProfileSmry.Load("scan-ph-tri", startswith="cvd_1_rtx_1")
    ps[9] = ProfileSmry.FromExtrapolation( ps[0].npho,  time_for_1M=100. )


