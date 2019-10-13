#!/usr/bin/env python

from __future__ import print_function
import os, sys, logging, numpy as np, argparse
from collections import OrderedDict as odict
log = logging.getLogger(__name__)
from opticks.ana.profilesmry import ProfileSmry


class ProfileSmryTab(object):
    def idx(self, pfx, cat):
        """
        :param pfx: string
        :param cat: string
        :return idx: index of the pair 
        """
        npfx = len(self.pfxs)
        ncat = len(self.cats)

        assert pfx in self.pfxs
        assert cat in self.cats
        ipfx = self.pfxs.index(pfx)
        icat = self.cats.index(cat)
        return ipfx*ncat + icat 

    def pfxcat(self, idx):
        """
        :param idx:
        :return pfx, cat:
        """
        npfx = len(self.pfxs)
        ncat = len(self.cats)
        icat = ( idx % ncat ) 
        ipfx = ( idx - icat ) / ncat
        return self.pfxs[ipfx], self.cats[icat]

    def __init__(self, pfxs, cats, gpufallback):
        self.pfxs = pfxs
        self.cats = cats
        ps = odict()  
        for pfx in pfxs:
            for cat in cats:
                idx = self.idx(pfx, cat) 
                pfx2, cat2 = self.pfxcat(idx) 
                assert pfx2 == pfx  
                assert cat2 == cat  
                print(" %5s %5s -> %2d   " % ( pfx, cat, idx )) 

                ps[idx] = ProfileSmry.Load(pfx, startswith=cat, gpufallback=gpufallback )
            pass
        pass
        assert len(ps) < 9 and len(ps) > 0, len(ps) 
        ps[9] = ProfileSmry.FromExtrapolation( ps[0].npho, time_for_1M=239. )
        self.ps = ps


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO) 

     pfxs = "scan-ph-8 scan-ph-9".split()

     cvd = "1"
     cats = "cvd_%(cvd)s_rtx_0 cvd_%(cvd)s_rtx_1" % locals() 
     cats = cats.split()  

     pst = ProfileSmryTab(pfxs, cats)

 




