#!/usr/bin/env python
"""
bench.py
============

Presents launchAVG times and prelaunch times for groups of Opticks runs
with filtering based on commandline arguments of the runs and the digest 
of the geocache used.

::

    bench.py --include xanalytic --digest f6cc352e44243f8fa536ab483ad390ce
    bench.py --include xanalytic --digest f6
        selecting analytic results for a particular geometry 

    bench.py --include xanalytic --digest 52e --since May22_1030
        selecting analytic results for a particular geometry after some time 

    bench.py --digest 52 --since 6pm


"""
import os, re, logging, sys, argparse
import numpy as np
log = logging.getLogger(__name__)

from dateutil.parser import parse
from datetime import datetime
from opticks.ana.datedfolder import DatedFolder, dateparser
from opticks.ana.meta import Meta
from opticks.ana.geocache import keydir

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument(     "--resultsdir", default="$TMP/results", help="Directory path to results" )
    parser.add_argument(     "--name",    default="geocache-bench", help="Name of directory beneath resultsdir in which to look for results")
    parser.add_argument(     "--digest",    default=None, help="Select result groups using geocaches with digests that start with the option string")
    parser.add_argument(     "--since",    default=None, help="Select results from dated folders following the date string provided, eg May22_1030 or 20190522_173746")
    parser.add_argument(     "--include", default=None, help="Select result groups with commandline containing the string provided" )
    parser.add_argument(     "--exclude", default=None, help="Select result groupd with commandline not containing the string provided" )
    parser.add_argument(     "--metric", default="launchAVG" );
    parser.add_argument(     "--other", default="prelaunch000" );
    args = parser.parse_args()

    print(args)
    base = os.path.join( args.resultsdir, args.name )
    base = os.path.expandvars(base)

    if args.since is not None:
        now = datetime.now()
        default = datetime(now.year, now.month, now.day)
        since = parse(args.since.replace("_"," "), default=default)  
        print("since : %s " % since )
    else:
        since = None
    pass 


    dirs, dfolds, dtimes = DatedFolder.find(base)
    metric = args.metric 
    other = args.other  


    ## arrange into groups of runs with the same runstamp/datedfolder
    assert len(dfolds) == len(dtimes) 
    order = sorted(range(len(dfolds)), key=lambda i:dtimes[i])

    for i in order:

        df = dfolds[i] 
        dt = dtimes[i] 

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
        rowfmt_ = lambda row:" %30s %10.3f %10.3f %10.3f      %10.3f   " % ( row.label, row.metric, row.rfast, row.rslow, row.other )

        lab = ( df,metric,"rfast", "rslow", other)

        metric_ = lambda m:float(m.d["OTracerTimes"][metric])
        other_ = lambda m:float(m.d["OTracerTimes"][other])
        key_ = lambda m:m.d["parameters"]["OPTICKS_KEY"]

        smm = sorted(mm, key=metric_)  
        ffast = metric_(smm[0])
        fslow = metric_(smm[-1])

        keys = map(key_, smm)
        assert len(set(keys)) == 1, "all OPTICKS_KEY for a group of runs with same dated folder should be identical " 
        key = keys[0]
        digest = key.split(".")[-1]
        idpath = keydir(key) 

 
        cmdline = smm[0].d["parameters"]["CMDLINE"]

        if args.include is not None and cmdline.find(args.include) == -1:
            continue   
        elif args.exclude is not None and cmdline.find(args.exclude) > -1:
            continue   
        elif args.digest is not None and not digest.startswith(args.digest):
            continue   
        elif since is not None and not dt > since:
            continue
        else:
            pass
        pass


        print("\n---")
        print(cmdline)
        print(key)
        print(idpath)
        print(labfmt_(lab))

        for i, m in enumerate(smm):

            f = metric_(m)
            rfast = f/ffast
            rslow = f/fslow
            o = other_(m)
               
            a[i] = (i, m.parentfold, f, rfast, rslow, o )  

            print(" %s : %s  " % ( rowfmt_(a[i]), m.absdir) )
        pass
    pass 

    print(args)






    
