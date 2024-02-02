#!/usr/bin/env python
"""
sprof.py
======================

::

   ~/o/sysrap/tests/sprof.sh 


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

if __name__ == '__main__':
    fold = Fold.Load("$SPROF_FOLD", symbol="fold")
    print(repr(fold))
    rp = fold.SEvt__EndOfRun_SProf_txt.reshape(-1,7,3)
pass



