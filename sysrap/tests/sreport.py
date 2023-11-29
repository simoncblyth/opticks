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


def make_title(meta, method):
    base = meta.base.replace("/data/blyth/opticks/GEOM/", "")
    smry = meta.smry("GPUMeta,prefix,creator")
    sfmt = meta.smry("stampFmt") 
    titl = "%s:%s %s " % (symbol,method, sfmt) 
    title = " ".join([titl,base,smry]) 
    return title

class Stamps(object):
    def __init__(self, f, symbol="A"):

        s = f.substamp
        meta = f.substamp_meta
        names = f.substamp_names


        assert len(s.shape) == 2 

        title = make_title(meta, method="Stamps")

        #e_sel = slice(1,None)              # skip 1st event, as initialization messes timings
        #t_sel = slice(2,None)              # skip first two stamps (init, BeginOfRun) 
        ## TODO: arrange stamps to avoid these 
        e_sel = slice(None)
        t_sel = slice(None)
        

        e_rel = names[e_sel]      # rel path of the evt folds, eg shape (9,)
        t_lab = labels_(f.labels)[t_sel]

        smry_ = lambda _:NPMeta.Summarize(_.decode("utf-8"))
        s_lab = list(map(smry_, t_lab))

        hdr = (" " * 8  + " %4s " * len(s_lab) ) % tuple(s_lab) 


        ss =  s[e_sel,t_sel]        # selected timestamps, eg shape (9,13)

        dss = ss - ss[:,0,np.newaxis]      # subtract first column stamp from all stamps row by row
                                           # hence giving begin of event relative time delta in microseconds

        assert dss.shape == ss.shape 

        assert len(e_rel) == dss.shape[0]  # event dimension 
        assert len(t_lab) == dss.shape[1]  # time stamp dimension 

        self.ss  = ss 
        self.dss = dss 
        self.s_lab = s_lab
        self.hdr = hdr
        self.title = title
        self.symbol = symbol
 
    def __repr__(self):
        return "\n".join([self.title, self.hdr, "%s.dss" % self.symbol, repr(self.dss)])

    def plot(self):
        st = self
        ax = None
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=st.title, equal=False)
            ax = axs[0]

            if TLIM[1] > TLIM[0]:
                ax.set_xlim(*TLIM)
            pass

            dss = st.dss
            for i in range(len(dss)):
                for j in range(len(dss[i])):
                    label = None if i > 0 else st.s_lab[j]
                    color = palette[j % len(palette)]
                    ax.vlines( dss[i,j], i-0.5, i+0.5, label=label , colors=[color] ) 
                pass
            pass
            ax.legend(loc="center")
            fig.show()
        pass  
        return ax


if __name__ == '__main__':
    ab = Fold.Load(symbol="ab")
    print(repr(ab))
    print("MODE:%d" % MODE) 

    if PICK in ["AB", "BA", "A", "B"]:
        for symbol in PICK:
            sym = symbol.lower()  
            f = getattr(ab, sym, None)
            if f is None: 
                print("sym:%s MISSING " % sym)
                continue 
            pass 
            st = Stamps(f, symbol=symbol)
            print(repr(st))
            ax = st.plot()
        pass
    pass


