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

            prof = Profile(os.path.dirname(path), name)
            prof.npho = npho 

            print("%20s %s " % (cat, path))
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

        for i, kv in enumerate(sorted(s.items(), key=lambda kv:kv[1].npho )): 
            launch[i] = kv[1].launch
            alaunch[i] = np.average( launch[i][1:] )
            interval[i] = kv[1].start_interval
            ainterval[i] = np.average( interval[i][1:] )
            npho[i] = kv[1].npho
        pass

        ps = cls(s)
        ps.launch = launch  
        ps.alaunch = alaunch  
        ps.interval = interval 
        ps.ainterval = ainterval 
        ps.npho = npho
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
    def FromAB(cls, a, b, att="ainterval" ):
        s = odict()
        ps = cls(s)

        assert np.all( a.npho == b.npho )

        ps.npho = a.npho
        ps.ratio = getattr(b,att)/getattr(a, att)
        return ps  

    def __init__(self, s):
        self.s = s 

    def __repr__(self):
        return "\n".join(["ProfileSmry", Profile.Labels()]+map(lambda kv:repr(kv[1]), sorted(self.s.items(), key=lambda kv:kv[1].npho )  ))



class O(object):
    ID = "scan-ph (profilesmry.py)"
    TT = odict()

    @classmethod
    def Get(cls, key):
        try:
            ikey = int(key)
            o = None if ikey < 0 else cls.TT.values()[ikey]
        except ValueError:
            o = cls.TT.get(key)
        pass
        return o 

    def __init__(self, key, desc):
        self.TT[key] = self
        self.key = key
        self.desc = desc
        self.title = "%s : %s " % ( key, desc )
        self.ratio = desc.find("Ratio") > -1  
        self.cfg4 = desc.find("G4") > -1 

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

        self.xlabel = "Number of Photons (1M to 100M)"
        self.ylabel = "Times (s)" if not self.ratio else "Ratio"

        self.ylim = None
        self.xlog = False
        self.ylog = False
        self.loc = "upper left"

        if self.key == "RTX_Speedup":
            self.ylim = [0,10]          
        elif self.key == "Opticks_vs_Geant4":
            self.ylog = True
            self.loc = [0.6, 0.5 ] 
        elif self.ratio and self.cfg4:
            self.ylim = [0, 55000]  
            self.ylog = False
        else:
            self.loc = [0.7, 0.6 ] 
            pass
        pass


def make_fig( plt, o, ps, rs ):

    plt.rcParams['figure.figsize'] = o.figsize    
    plt.rcParams['font.size'] = o.fontsize

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(o.title, fontsize=o.fontsize)

    if not o.ratio:
        ii = [9, 0, 1 ] if o.cfg4 else [0,1 ]
        for i in ii:
            p = ps[i]
            g4 = p.label[:2] == "G4"
            if not g4:
                plt.plot( p.npho, p.ainterval, p.fmt, c="b", label="%s (interval)" % p.label )
                plt.plot( p.npho, p.alaunch,   p.fmt, c="r", label="%s (launch)" % p.label )
            else:
                plt.plot( p.npho, p.ainterval, p.fmt, c="g", label="%s" % p.label )
            pass
        pass
    pass

    if o.ratio: 
        rr = "09 19" if o.cfg4 else "10"
        for j in rr.split():
            r = rs[j]
            plt.plot( r.npho,  r.ratio,  r.fmt, label=r.label  ) 
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
    ax.legend(loc=o.loc, fontsize=o.fontsize)
    return fig 






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=200)

    O("Opticks_vs_Geant4",  "Extrapolated G4 times compared to Opticks launch+interval times with RTX mode ON and OFF")  
    O("Opticks_Speedup",    "Ratio of extrapolated G4 times to Opticks launch+interval times with RTX mode ON and OFF")  
    O("Overheads",   "Comparison of Opticks GPU launch times and intervals with RTX mode ON and OFF")  
    O("RTX_Speedup", "Ratio of launch times with RTX mode OFF to ON ")  

    o = O.Get(-1)

    ps = {}

    ps[0] = ProfileSmry.Load("scan-ph", startswith="cvd_1_rtx_0")
    ps[1] = ProfileSmry.Load("scan-ph", startswith="cvd_1_rtx_1")
    ps[9] = ProfileSmry.FromExtrapolation( ps[0].npho,  time_for_1M=100. )

    ps[0].fmt = "o:"  
    ps[0].label = "TITAN RTX, RTX OFF"

    ps[1].fmt = "o-"
    ps[1].label = "TITAN RTX, RTX ON"

    ps[9].fmt = "o--" 
    ps[9].label = "G4 Extrapolated"



    import matplotlib.pyplot as plt
    plt.ion()

    oo = O.TT.values() if o is None else [o] 
    for o in oo:

        rs = {}
        if o.ratio:
            if o.cfg4: 
                rs["09"] = ProfileSmry.FromAB( ps[0], ps[9] )
                rs["09"].fmt = "ro--"
                rs["09"].label = "G4 Extrapolated / TITAN RTX, RTX OFF (interval)"

                rs["19"] = ProfileSmry.FromAB( ps[1], ps[9] )
                rs["19"].fmt = "bo--"
                rs["19"].label = "G4 Extrapolated / TITAN RTX, RTX ON (interval)"
            else:
                rs["10"] = ProfileSmry.FromAB( ps[1], ps[0], att="alaunch" )
                rs["10"].fmt = "ro--"
                rs["10"].label = "TITAN RTX, RTX OFF/ON (launch)"
            pass 
        pass

        fig = make_fig( plt, o, ps, rs )
        fig.show()

        plt.savefig(o.path)

    pass

  
