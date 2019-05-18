#!/usr/bin/env python
"""

"""
import os, re, logging, sys, argparse
import numpy as np
log = logging.getLogger(__name__)

from opticks.ana.datedfolder import DatedFolder, dateparser
from opticks.ana.meta import Meta

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(    "base", default=None, help="Directory below which to look for results")
    parser.add_argument(     "--include", default=None, help="Select result groups with commandline containing the string provided" )
    parser.add_argument(     "--exclude", default=None, help="Select result groupd with commandline not containing the string provided" )
    parser.add_argument(     "--metric", default="launchAVG" );
    parser.add_argument(     "--other", default="prelaunch000" );
    args = parser.parse_args()

    print(args)

    base = args.base if args.base is not None else "."
    print(base)

    dirs, dfolds = DatedFolder.find(base)
    metric = args.metric 
    other = args.other  

    for df in sorted(dfolds):
        udirs = filter(lambda _:_.endswith(df),dirs)
        #print("\n".join(udirs))


        mm = [Meta(p, base) for p in udirs]

        dtype = [ 
              ("index", np.int32),
              ("label", "|S30"),
              ("metric", np.float32),
              ("rfast", np.float32),
              ("rslow", np.float32),
              ("other", np.float32),
                ]

        a = np.recarray((len(mm),), dtype=dtype )

        labfmt_ = lambda lab:" %30s %10s %10s %10s      %10s " %  lab
        rowfmt_ = lambda row:" %30s %10.3f %10.3f %10.3f      %10.3f " % ( row.label, row.metric, row.rfast, row.rslow, row.other )

        lab = ( df,metric,"rfast", "rslow", other)

        metric_ = lambda m:float(m.d["OTracerTimes"][metric])
        other_ = lambda m:float(m.d["OTracerTimes"][other])


        smm = sorted(mm, key=metric_)  
        ffast = metric_(smm[0])
        fslow = metric_(smm[-1])

        cmdline = smm[0].d["parameters"]["CMDLINE"]

        if args.include is not None and cmdline.find(args.include) == -1:
            continue   
        elif args.exclude is not None and cmdline.find(args.exclude) > -1:
            continue   
        else:
            pass
        pass


        print(cmdline)
        print(labfmt_(lab))
        for i, m in enumerate(smm):

            f = metric_(m)
            rfast = f/ffast
            rslow = f/fslow
            o = other_(m)
               
            a[i] = (i, m.parentfold, f, rfast, rslow, o )  

            print(rowfmt_(a[i]))
        pass
    pass 







    
