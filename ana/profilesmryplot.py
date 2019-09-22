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
profilesmryplot.py
====================

::

    an ; ip profilesmryplot.py

    an ; ip profilesmryplot.py --pfx "scan-ph" --gpu "TITAN V" --cvd 1 
    an ; ip profilesmryplot.py --pfx "scan-ph-0" --gpu Quadro_RTX_8000 --cvd 0 
    an ; ip profilesmryplot.py --pfx "scan-ph-1" --gpu Quadro_RTX_8000 --cvd 0 
    an ; ip profilesmryplot.py --pfx "scan-ph-2" --gpu Quadro_RTX_8000 --cvd 0 
    an ; ip profilesmryplot.py --pfx "scan-ph-3" --gpu Quadro_RTX_8000 --cvd 0 
    an ; ip profilesmryplot.py --pfx "scan-ph-4" --gpu TITAN_RTX --cvd 1 
    an ; ip profilesmryplot.py --pfx "scan-ph-5" --gpu TITAN_RTX --cvd 1 

    an ; ip profilesmryplot.py --pfx "scan-ph-7" --gpu TITAN_RTX --cvd 1 
    an ; ip profilesmryplot.py --pfx "scan-ph-8" --gpu TITAN_RTX --cvd 1 
    an ; ip profilesmryplot.py --pfx "scan-ph-9" --gpu TITAN_RTX --cvd 1 
    an ; ip profilesmryplot.py --pfx "scan-ph-10" --gpu TITAN_RTX --cvd 1 

"""

from __future__ import print_function
import os, sys, logging, numpy as np, argparse
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

from opticks.ana.num import Num
from opticks.ana.base import findfile
from opticks.ana.profile import Profile
from opticks.ana.profilesmry import ProfileSmry, ProfileMain
from opticks.ana.profilesmrytab import ProfileSmryTab


class O(object):
    """

    """
    ID = "scan-ph (profilesmryplot.py)"
    TT = odict()
    SLI = slice(0,None)

    @classmethod
    def Get(cls, key):
        try:
            ikey = int(key)
            o = None if ikey < 0 else cls.TT.values()[ikey]
        except ValueError:
            o = cls.TT.get(key)
        pass
        return o 

    def __repr__(self):
        return " key %s ratio %s cfg4 %s " % (self.key, self.ratio, self.cfg4 ) 

    @classmethod
    def XLabel(cls, sli):
        mxph = 100 if sli.stop is None else (sli.stop - 1)*10   
        return "Number of Photons (1M to %dM)" % mxph

    def _get_xlabel(self):
        return self.XLabel(self.sli)
    xlabel = property(_get_xlabel)


    def __init__(self, key, desc):
        self.TT[key] = self
        self.key = key
        self.desc = desc
        self.title = "%s : %s " % ( key, desc )
        self.ratio = desc.find("Ratio") > -1  
        self.cfg4 = desc.find("G4") > -1 
        self.sli = self.SLI

        #self.fontsize = 20
        #self.figsize = 18,10.2  # png 1800,1019 pixels fontsize 20 looks good 
        self.fontsize = 15  
        self.figsize = 12.8,7.20   # png of 1280,720   fontsize 20 too big 

        self.path = os.path.expandvars(os.path.join("$TMP/ana", "%s.png" % self.key )) 
        self.dir_ = os.path.dirname(self.path)
        if not os.path.exists(self.dir_):
            log.info("creating dir %s " % self.dir_)
            os.makedirs(self.dir_)   
        pass

        self.ylabel = "Times (s)" if not self.ratio else "Ratio"

        self.ylim = None
        self.xlog = False
        self.ylog = False
        self.loc = "upper left"

        if self.ratio:
            if self.cfg4:
                if self.key == "Opticks_Speedup":
                    self.ylim = [0, 20000]  
                    self.rr = "19l 19i 09l 09i"
                    self.ylog = False
                    self.loc = [0.4, 0.45 ] 
                else:
                    assert 0, self.key 
                pass
            else:  
                if self.key == "RTX_Speedup":
                    self.rr = "10" 
                    self.ylim = [0,10]          
                elif self.key == "Interval_over_Launch":
                    self.ylim = [0, 5]  
                    self.rr = "00 11" 
                    self.loc = "upper right"   
                else:
                    assert 0, self.key 
                pass
            pass
        else:
            if self.cfg4:
               self.ii = [9,0,1]
            else:   
               self.ii = [0,1]
            pass

            if self.key == "Overheads":
                self.loc = "upper left"   
            elif self.key == "NHit":
                self.ii = [1]
                self.sli = slice(None)
                self.loc = "upper left"   
                self.ylabel = "Number of Hits " 
            elif self.key == "Opticks_vs_Geant4":
                self.ylog = True
                self.loc = [0.6, 0.5 ] 
            else:
                pass
            pass

        pass


def make_fig( plt, o, ps, rs ):

    sli = o.sli
    plt.rcParams['figure.figsize'] = o.figsize    
    plt.rcParams['font.size'] = o.fontsize

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(o.title, fontsize=o.fontsize)

    if not o.ratio:
        for i in o.ii:
            p = ps[i]
            g4 = p.label[:2] == "G4"
            if not g4:
                if i == 1 and o.key == "NHit":
                    plt.plot( p.npho[sli], p.nhit[sli],   p.fmt, c="r", label="%s NHit" % p.label )
                else:
                    plt.plot( p.npho[sli], p.ainterval[sli], p.fmt, c="b", label="%s (interval)" % p.label )
                    plt.plot( p.npho[sli], p.alaunch[sli],   p.fmt, c="r", label="%s (launch)" % p.label )
                pass
            else:
                plt.plot( p.npho[sli], p.ainterval[sli], p.fmt, c="g", label="%s" % p.label )
            pass
        pass
    pass

    if o.ratio: 
        for j in o.rr.split():
            r = rs[j]
            print(j)
            plt.plot( r.npho[sli],  r.ratio[sli],  r.fmt, label=r.label  ) 
        pass
    pass

    if not o.ylabel is None:
        plt.ylabel(o.ylabel, fontsize=o.fontsize)
    pass
    if not o.xlabel is None:
        plt.xlabel(o.xlabel, fontsize=o.fontsize)
    pass
    if o.ylog:
        ax.set_yscale('log')
    pass
    if o.xlog:
        ax.set_xscale('log')
    pass
    if not o.ylim is None:
        ax.set_ylim(o.ylim)
    pass
    ax.legend(loc=o.loc, fontsize=o.fontsize, shadow=True)
    return fig 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=200)

    pm = ProfileMain.ParseArgs(__doc__) 

    # plot specifications 

    O("Opticks_vs_Geant4",  "Extrapolated G4 times compared to Opticks launch+interval times with RTX mode ON and OFF")  
    O("Opticks_Speedup",    "Ratio of extrapolated G4 times to Opticks launch(interval) times")  
    O("Overheads",   "Comparison of Opticks GPU launch times and intervals with RTX mode ON and OFF")  
    O("RTX_Speedup", "Ratio of launch times with RTX mode OFF to ON ")  
    O("Interval_over_Launch",   "Ratio of Opticks GPU interval/launch times with RTX mode ON and OFF").comment = "RTX ON overheads are worse because launch times reduced"  
    O("NHit",        "Number of Hits vs Number of Photons")

    o = O.Get(-1)  # None for -1, corresponding to all plots OR a single plot specification selected by index

    pfx = pm.pfx0
    cvd = pm.cvd

    cat0 = "cvd_%s_rtx_0" % cvd 
    cat1 = "cvd_%s_rtx_1" % cvd

    pfxs = [pfx,]
    cats = [cat0, cat1]


    pftab = ProfileSmryTab(pfxs, cats, pm.gpufallback)
    ps = pftab.ps


    ps[0].fmt = "o:"  
    ps[0].label = ps[0].autolabel

    ps[1].fmt = "o-"
    ps[1].label = ps[1].autolabel

    assert ps[0].gpu == ps[1].gpu, (ps[0].gpu, ps[1].gpu) 
    gpu = ps[0].gpu 

    ps[9].fmt = "o--" 
    ps[9].label = "G4 Extrapolated"


    import matplotlib.pyplot as plt
    plt.ion()

    oo = O.TT.values() if o is None else [o] 
    for o in oo:

        print(o)
  
        rs = {}
        if o.ratio:
            if o.cfg4: 
                rs["09l"] = ProfileSmry.FromAB( ps[0], ps[9], att="alaunch" )
                rs["09l"].fmt = "ro--"
                rs["09l"].label = "G4 Extrapolated / %s (launch)" % ps[0].label

                rs["09i"] = ProfileSmry.FromAB( ps[0], ps[9], att="ainterval" )
                rs["09i"].fmt = "bo--"
                rs["09i"].label = "G4 Extrapolated / %s (interval)" % ps[0].label

                rs["19l"] = ProfileSmry.FromAB( ps[1], ps[9], att="alaunch")
                rs["19l"].fmt = "ro-"
                rs["19l"].label = "G4 Extrapolated / %s (launch)" % ps[1].label

                rs["19i"] = ProfileSmry.FromAB( ps[1], ps[9], att="ainterval")
                rs["19i"].fmt = "bo-"
                rs["19i"].label = "G4 Extrapolated / %s (interval)" % ps[1].label 

            else:


                rs["10"] = ProfileSmry.FromAB( ps[1], ps[0], att="alaunch" )
                rs["10"].fmt = "ro--"
                rs["10"].label = "%s, RTX OFF/ON (launch)" % gpu

                rs["00"] = ProfileSmry.FromAtt( ps[0], num_att="ainterval", den_att="alaunch" )
                rs["00"].fmt = "bo--"
                rs["00"].label = "%s  interval / launch " % ps[0].label

                rs["11"] = ProfileSmry.FromAtt( ps[1], num_att="ainterval", den_att="alaunch" )
                rs["11"].fmt = "ro--"
                rs["11"].label = "%s interval / launch " % ps[1].label 
            pass 
        pass

        fig = make_fig( plt, o, ps, rs )
        fig.show()

        plt.savefig(o.path)

    pass

  
