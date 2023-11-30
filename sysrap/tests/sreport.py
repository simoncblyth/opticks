#!/usr/bin/env python
"""
sreport.py
======================

TODO: incorporate sprof_fold_report.py into this

"""

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.npmeta import NPMeta

MODE =  int(os.environ.get("MODE", "2"))
PICK =  os.environ.get("PICK", "AB")
TLIM =  np.array(list(map(int,os.environ.get("TLIM", "0,0").split(","))),dtype=np.int32)


if MODE != 0:
    from opticks.ana.pvplt import * 
pass

labels_ = lambda l:l.view("|S%d" % l.shape[1])[:,0]  
tv_     = lambda a:a.view("datetime64[us]")  

# https://matplotlib.org/stable/gallery/color/named_colors.html
palette = ["red","green", "blue", 
           "cyan", "magenta", "yellow", 
           "tab:orange", "tab:pink", "tab:olive",
           "tab:purple", "tab:grey", "tab:cyan"
           ]


def make_title(meta, method, symbol):
    base = meta.base.replace("/data/blyth/opticks/GEOM/", "")
    smry = meta.smry("GPUMeta,prefix,creator")
    sfmt = meta.smry("stampFmt") 
    titl = "%s:%s %s " % (symbol,method, sfmt) 
    title = " ".join([titl,base,smry]) 
    return title

smry__ = lambda _:NPMeta.Summarize(_)
smry_ = lambda _:list(map(smry__, _))

class Substamp(object):
    def __init__(self, f, symbol="fold.substamp.a"):

        substamp = f.substamp
        meta = f.substamp_meta
        names = f.substamp_names
        delta = f.delta_substamp 
        labels = f.substamp_labels
        etime = f.delta_substamp[:,-1]  

        _icol = np.where(f.subcount_labels == 'photon')[0] 
        icol = _icol[0] if len(_icol) == 1 else -1 
        subcount_photon = f.subcount[:,icol] if icol > -1 else None

        labels_s = smry_(labels)
        hdr = (" " * 8  + " %4s " * len(labels_s) ) % tuple(labels_s) 

        title = make_title(meta, method="Substamp", symbol=symbol)

        assert len(substamp.shape) == 2 
        assert delta.shape == substamp.shape 

        self.substamp = substamp
        self.delta = delta 
        self.labels_s = labels_s
        self.hdr = hdr
        self.title = title
        self.symbol = symbol

        self.etime = etime 
        self.subcount_photon = subcount_photon
 
    def __repr__(self):
        return "\n".join([self.title, self.hdr, self.symbol, repr(self.delta)])

    def plot_etime_vs_photon(self):
        ss = self
        ax = None
        etime = self.etime
        subcount_photon = self.subcount_photon

        if etime is None:
            log.error("plot_etime_vs_photon.ABORT etime None")
            return 
        pass
        if subcount_photon is None:
            log.error("plot_etime_vs_photon.ABORT subcount_photon None")
            return 
        pass
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=ss.title, equal=False)
            ax = axs[0]
            ax.scatter( subcount_photon, etime, label="etime_vs_photon")
            ax.legend()
            fig.show()
        pass  
        return ax


    def plot_delta(self):
        ss = self
        ax = None
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=ss.title, equal=False)
            ax = axs[0]
            if TLIM[1] > TLIM[0]:
                ax.set_xlim(*TLIM)
            pass
            dss = ss.delta
            for i in range(len(dss)):
                for j in range(len(dss[i])):
                    label = None if i > 0 else ss.labels_s[j]
                    color = palette[j % len(palette)]
                    ax.vlines( dss[i,j], i-0.5, i+0.5, label=label , colors=[color] ) 
                pass
            pass
            ax.legend(loc="center")
            fig.show()
        pass  
        return ax


if __name__ == '__main__':
    fold = Fold.Load(symbol="fold")

    print(repr(fold))
    print("MODE:%d PICK:%s " % (MODE, PICK) ) 

    if hasattr(fold, "substamp"):
        for e in PICK:
            f = getattr(fold.substamp, e.lower(), None)
            symbol = "fold.substamp.%s" % e.lower() 
            if f is None: 
                print("%s : MISSING " % symbol)
                continue 
            pass 
            ss = Substamp(f, symbol=symbol )
            print(repr(ss))
            ax = ss.plot_delta()
            ax = ss.plot_etime_vs_photon()
        pass
    pass


