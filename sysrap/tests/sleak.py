#!/usr/bin/env python
"""
sleak.py
======================

::

   ~/o/sysrap/tests/sleak.sh 
   DRM=1 ~/o/sysrap/tests/sleak.sh
   DRM=1 YLIM=0,3 ~/o/sysrap/tests/sleak.sh ana 


"""

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.npmeta import NPMeta

COMMANDLINE = os.environ.get("COMMANDLINE", "")
STEM =  os.environ.get("STEM", "")
HEADLINE = "%s ## %s " % (COMMANDLINE, STEM ) 
JOB =  os.environ.get("JOB", "")
PLOT =  os.environ.get("PLOT", "Runprof_ALL")
STEM =  os.environ.get("STEM", "")
PICK =  os.environ.get("PICK", "AB")
TLIM =  np.array(list(map(int,os.environ.get("TLIM", "0,0").split(","))),dtype=np.int32)
YLIM = np.array(list(map(float, os.environ.get("YLIM","0,0").split(","))),dtype=np.float32)
 

MODE =  int(os.environ.get("MODE", "2"))

if MODE != 0:
    from opticks.ana.pvplt import * 
pass



class Subprofile(object):
    """
    Why VM is so large with CUDA

    * https://forums.developer.nvidia.com/t/high-virtual-memory-consumption-on-linux-for-cuda-programs-is-it-possible-to-avoid-it/67706/4

    """
    FONTSIZE = 20 
    XLABEL = "Time from 1st sprof.h stamp (seconds)"
    YLABEL_VM = "sprof.h VM memory (GB) "
    YLABEL_RS = "sprof.h RSS memory (GB) "
    YLABEL_DRM = "sprof.h DRM per event leak (MB) "



class Runprof_ALL(object):
    """
    PLOT=Runprof_ALL ~/o/sysrap/tests/sleak.sh 
    """
    def __init__(self, fold, symbol="fold.runprof"):
        rp = eval(symbol)
        self.rp = rp
        label = "Runprof_ALL " 
        fontsize = Subprofile.FONTSIZE
        if MODE == 2 or MODE == 3:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
            ax = axs[0]

            tp = (rp[:,0] - rp[0,0])/1e6  # seconds
            vm = rp[:,1]/1e6              # GB
            rs = rp[:,2]/1e6              # GB

            #tpm = tp[0::2]
            #drm = (rp[1::2,2] - rp[0::2,2])/1e3  # MB 
            
            tpm = tp[2::2]
            drm = np.diff(rp[0::2,2])/1e3  # MB 
            slm = slice(1,None)
            tpm = tpm[slm]
            drm = drm[slm]

            self.tp = tp
            self.vm = vm
            self.rs = rs
            self.drm = drm 
            self.tpm = tpm 

            if "DRM" in os.environ:
                label = "%s : %s vs starttime(s)" % (symbol, Subprofile.YLABEL_DRM)
                ax.scatter( tpm, drm, label=label )
                ax.plot(    tpm, drm )
                ax.set_ylabel(Subprofile.YLABEL_DRM, fontsize=Subprofile.FONTSIZE )
                pass
            elif "VM" in os.environ:
                label = "%s : %s vs time(s)" % (symbol, Subprofile.YLABEL_VM)
                ax.scatter( tp, vm, label=label )
                ax.plot(    tp, vm )
            else:
                label = "%s : %s vs time(s)" % (symbol, Subprofile.YLABEL_RS)
                ax.scatter( tp, rs, label=label  )
                ax.plot( tp, rs )
            pass
            ax.set_xlabel(Subprofile.XLABEL, fontsize=Subprofile.FONTSIZE )
            if YLIM[1] > YLIM[0]:
                ax.set_ylim(*YLIM)
            pass

            yl = ax.get_ylim()
            ## THIS ASSUMES A AND B SEVT STAMPS 
            #ax.vlines( tp[0::4], yl[0], yl[1], color="blue", label="nBeg" ) 
            #ax.vlines( tp[1::4], yl[0], yl[1], color="red", label="nEnd" ) 
            #ax.vlines( tp[2::4], yl[0], yl[1], color="cyan", label="pBeg" ) 
            #ax.vlines( tp[3::4], yl[0], yl[1], color="pink", label="pEnd" ) 

            if "DRM" in os.environ:
                pass
            else:
                ax.vlines( tp[0::2], yl[0], yl[1], color="blue", label="ABeg" ) 
                ax.vlines( tp[1::2], yl[0], yl[1], color="red", label="AEnd" ) 
            pass

            ax.legend()
            fig.show()
        else:
            print("MODE:%d " % MODE)
        pass  



if __name__ == '__main__':
    fold = Fold.Load("$SLEAK_FOLD", symbol="fold")
    print("SLEAK_FOLD:%s " % os.environ["SLEAK_FOLD"] )
    print(repr(fold))
    if PLOT.startswith("Runprof_ALL") and hasattr(fold, "runprof"):
        RPA = Runprof_ALL(fold, symbol="fold.runprof" )  ## RSS vs time : profile plot 
        rp = RPA.rp
        drm = RPA.drm
    else:
        print("PLOT:%s UNHANDLED" % PLOT)
    pass
pass

