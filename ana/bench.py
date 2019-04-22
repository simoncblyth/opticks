#!/usr/bin/env python
"""

"""
import os, re, logging, sys
import numpy as np
log = logging.getLogger(__name__)

from opticks.ana.datedfolder import DatedFolder, dateparser
from opticks.ana.meta import Meta

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     base = sys.argv[1] if len(sys.argv) > 1 else "." 
     dirs, dfolds = DatedFolder.find(base)


     for df in sorted(dfolds):
         udirs = filter(lambda _:_.endswith(df),dirs)
         #print("\n".join(udirs))


         mm = [Meta(p, base) for p in udirs]

         dtype = [ 
              ("index", np.int32),
              ("label", "|S30"),
              ("metric", np.float32),
              ("rfast", np.float32),
              ("rslow", np.float32)
                ]

         a = np.recarray((len(mm),), dtype=dtype )

         labfmt_ = lambda lab:" %30s %10s %10s %10s " % lab
         rowfmt_ = lambda row:" %30s %10.3f %10.3f %10.3f " % ( row.label, row.metric, row.rfast, row.rslow )

         lab = ( df,"metric","rfast", "rslow")

         metric_ = lambda m:float(m.d["OTracerTimes"]["launchAVG"])

         smm = sorted(mm, key=metric_)  
         ffast = metric_(smm[0])
         fslow = metric_(smm[-1])

         print(smm[0].d["parameters"]["CMDLINE"])

         print(labfmt_(lab))
         for i, m in enumerate(smm):

             f = metric_(m)
             rfast = f/ffast
             rslow = f/fslow
               
             a[i] = (i, m.parentfold, f, rfast, rslow )  

             print(rowfmt_(a[i]))
         pass
     pass 







    
