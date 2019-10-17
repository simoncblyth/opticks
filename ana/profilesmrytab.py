#!/usr/bin/env python
"""
profilesmrytab.py
===================

Loads and displays profile summaries, ie the results from scan-::

    [blyth@localhost ana]$ profilesmrytab.py scan-pf-1/cvd_0_rtx_1
    0
    ProfileSmry FromDict:scan-pf-1:cvd_0_rtx_1 /home/blyth/local/opticks/evtbase/scan-pf-1 0:Quadro_RTX_8000 
    0:Quadro_RTX_8000, RTX ON
              CDeviceBriefAll : 0:Quadro_RTX_8000 
              CDeviceBriefVis : 0:Quadro_RTX_8000 
                      RTXMode : 1 
        NVIDIA_DRIVER_VERSION : 435.21 
                     name       note  av.interv  av.launch  av.overhd :                                             launch :                                                  q 
           cvd_0_rtx_1_1M   MULTIEVT     0.1584     0.1324     1.1963 :   0.1367   0.1328   0.1328   0.1328   0.1289   0.1328   0.1328   0.1328   0.1328   0.1289 :   0.1373   0.1332   0.1329   0.1328   0.1324   0.1326   0.1325   0.1328   0.1328   0.1324 
          cvd_0_rtx_1_10M   MULTIEVT     1.4918     1.3062     1.1420 :   1.3086   1.3047   1.3047   1.3047   1.3086   1.3086   1.3047   1.3047   1.3047   1.3086 :   1.3086   1.3047   1.3044   1.3048   1.3055   1.3058   1.3052   1.3059   1.3061   1.3072 
          cvd_0_rtx_1_20M   MULTIEVT     2.8702     2.7578     1.0408 :   2.7578   2.7578   2.7578   2.7539   2.7617   2.7578   2.7578   2.7578   2.7578   2.7578 :   2.7578   2.7552   2.7577   2.7564   2.7587   2.7569   2.7586   2.7588   2.7602   2.7608 
          ...

"""

from __future__ import print_function
import os, sys, logging, numpy as np, argparse, textwrap
from collections import OrderedDict as odict
log = logging.getLogger(__name__)
from opticks.ana.profilesmry import ProfileSmry

class ProfileSmryTab(object):

    @classmethod
    def MakeCrossList(cls, pfxs, cats):
        pfxcats = [] 
        for pfx in pfxs:
            for cat in cats:
                pfxcat = "/".join([pfx, cat])
                pfxcats.append(pfxcat)
            pass
        pass
        return pfxcats          

    @classmethod
    def FromCrossList(cls, pfxs, cats):
        pfxcats=cls.MakeCrossList(pfxs, cats)
        return cls(pfxcats) 

    @classmethod
    def FromText(cls, txt):
        pfxcats=filter(None,textwrap.dedent(txt).split("\n"))
        return cls(pfxcats) 

    def __init__(self, pfxcats):
        ps = odict()  

        upfxs = []
        ucats = []

        for idx,pfxcat in enumerate(pfxcats):
            elem = pfxcat.split("/")
            assert len(elem) == 2  
            pfx, cat = elem
            if not pfx in upfxs:
                upfxs.append(pfx)
            pass
            if not cat in ucats:
                ucats.append(cat)
            pass
            ps[idx] = ProfileSmry.Load(pfx, startswith=cat, gpufallback=None )
        pass
        self.ps = ps
        self.pfxcats = pfxcats  
        self.upfxs = upfxs 
        self.ucats = ucats 

    def addG4Extrapolation(self, g4_seconds_1M=239.):
        ps = self.ps 
        assert len(ps) < 9 and len(ps) > 0, len(ps) 
        pass
        ps[9] = ProfileSmry.FromExtrapolation( ps[0].npho, seconds_1M=g4_seconds_1M )


    def idx(self, pfx, cat):
        pfxcat = "/".join([pfx,cat])
        return self.pfxcats.index(pfxcat)   

    def pfxcat(self, idx):
        pfxcat = self.pfxcats[idx]
        return pfxcat.split("/") 

    def __str__(self):
        return "\n".join(map(lambda kv:"%s\n%s\n" % (kv[0],kv[1]), self.ps.items()) + self.pfxcats)

    def __repr__(self):
        return "\n".join(["ProfileSmryTab"] + self.pfxcats)

    scanid = property(lambda self:"_".join(self.upfxs))  # eg scan-pf-0



def test_ph_8_9():
     pfxs = "scan-ph-8 scan-ph-9".split()
     cvd = "1"
     cats = "cvd_%(cvd)s_rtx_0 cvd_%(cvd)s_rtx_1" % locals() 
     cats = cats.split()  
     pst = ProfileSmryTab.FromCrossList(pfxs, cats)
     return pst 


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO) 

     import argparse
     parser = argparse.ArgumentParser(__doc__)
     parser.add_argument( "pfxcats", nargs="*", default=[], help="List of pfxcat to load and display eg scan-pf-0/cvd_1_rtx_0" )
     args = parser.parse_args()

     if len(args.pfxcats) == 0:
         pst = ProfileSmryTab.FromText("""
         scan-pf-0/cvd_1_rtx_0
         scan-pf-1/cvd_0_rtx_0
         scan-pf-0/cvd_1_rtx_1
         scan-pf-1/cvd_0_rtx_1
         """)
     else:
         pst = ProfileSmryTab(args.pfxcats)  
     pass
     print(pst)

 




