#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.npmeta import NPMeta
from opticks.sysrap.tests.sreport import Substamp 
from opticks.sysrap.tests.sreport import RUN_META 

MODE = 2
PLOT =  os.environ.get("PLOT","?plot") 
COMMANDLINE = os.environ.get("COMMANDLINE", "")


if MODE != 0:
    from opticks.ana.pvplt import * 
pass


class AB_Substamp_ALL_Etime_vs_Photon(object):
    def __init__(self, a, b):
        title = RUN_META.AB_Title(a,b)

        af = a.substamp.a
        at = Substamp.ETime(af)
        an = a.submeta_NumPhotonCollected.a  

        bf = b.substamp.a
        bt = Substamp.ETime(bf)
        bn = b.submeta_NumPhotonCollected.a  

        assert np.all( an == bn )
        photon = an[:,0]/1e6


        fontsize = 20
        YSCALE_ = ["log", "linear" ] 
        YSCALE = os.environ.get("YSCALE", YSCALE_[1])
        assert YSCALE in YSCALE_
  
        YMIN = float(os.environ.get("YMIN", "1e-2"))
        YMAX = float(os.environ.get("YMAX", "40"))

        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]


            sli = slice(15,20)

            deg = 1  # 1:lin 2:par 
            a_fit = np.poly1d(np.polyfit(photon[sli], at[sli], deg))
            a_fit_label = "%s : linefit( slope %10.3f  intercept %10.3f )" % (a.symbol, a_fit.coef[0], a_fit.coef[1])
            self.a_fit = a_fit

            b_fit = np.poly1d(np.polyfit(photon[sli], bt[sli], deg))
            b_fit_label = "%s : linefit( slope %10.3f  intercept %10.3f )" % (b.symbol, b_fit.coef[0], b_fit.coef[1])
            self.b_fit = b_fit

            a_label = RUN_META.GPUMeta(a)
            b_label = RUN_META.GPUMeta(b)

            ax.scatter( photon, at, label=a_label )
            ax.plot( photon[sli], a_fit(photon[sli]), label=a_fit_label )

            ax.scatter( photon, bt, label=b_label )
            ax.plot( photon[sli], b_fit(photon[sli]), label=b_fit_label )

            pass
            ax.set_xlim( -5, 105 ); 
            ax.set_ylim( YMIN, YMAX );  # 50*200 = 1e4
            ax.set_yscale(YSCALE)
            ax.set_ylabel("Event time (seconds)", fontsize=fontsize )
            ax.set_xlabel("Number of Photons (Millions)", fontsize=fontsize )
            ax.legend()
            ax.legend()
            fig.show()
        pass





if __name__ == '__main__':

    print("[sreport_ab.py:PLOT[%s]" % PLOT ) 

    asym = os.path.expandvars("$A")
    print("[sreport_ab.py:fold = Fold.Load A_SREPORT_FOLD [%s]" % asym ) 
    a = Fold.Load("$A_SREPORT_FOLD", symbol=asym )
    print("]sreport_ab.py:a = Fold.Load" ) 

    print("[sreport_ab.py:repr(a)" ) 
    print(repr(a))
    print("]sreport_ab.py:repr(a)" ) 


    bsym = os.path.expandvars("$B")
    print("[sreport_ab.py:fold = Fold.Load B_SREPORT_FOLD [%s]" % bsym ) 
    b = Fold.Load("$B_SREPORT_FOLD", symbol=bsym )
    print("]sreport_ab.py:b = Fold.Load" ) 

    print("[sreport_ab.py:repr(b)" ) 
    print(repr(b))
    print("]sreport_ab.py:repr(b)" ) 

    print("]sreport_ab.py:PLOT[%s]" % PLOT ) 


    if PLOT.startswith("AB_Substamp_ALL_Etime_vs_Photon"): 
        plt = AB_Substamp_ALL_Etime_vs_Photon(a,b)
    pass


    





