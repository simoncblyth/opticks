#!/usr/bin/env python

import os, sys, argparse, logging
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

class PTX(object):
    def __init__(self, path):
        lines = file(path).read().split("\n")
        d = odict()
        d["TOTAL"] = 0 
        region = "start"
        for i, line in enumerate(lines):
            if line.find(".entry") > -1:
                region = "%0.4d : %s " % ( i, line )  
                if region not in d:
                    d[region] = 0 
                pass
            pass                     
            f64 = line.find(".f64") > -1
            if f64:
               if self.exclude is not None and region.find(self.exclude) > -1:
                   pass
               else:
                   d[region] += 1   
                   d["TOTAL"] += 1   
               pass
            pass
        pass        
        self.path = path 
        self.d = d

    def __repr__(self):
        return "\n".join(["ptx.py %s" % self.path] + map(lambda kv:"%4d : %s " %  (kv[1],kv[0]), self.d.items()))
     

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "path", nargs='?', default=os.getcwd(), help="File or directory path with ptx" )
    parser.add_argument(     "--all", action="store_true", default=False, help="Report for all ptx, not just those with f64 " )
    parser.add_argument(     "--exclude", default=None, help="Dont count f64 from entry points containing the string provided" )
    parser.add_argument(     "--namefilter", default=None, help="String that the name of the ptx must include to be listed" )
    args = parser.parse_args()
    PTX.exclude = args.exclude
    cmdline = " ".join(["ptx.py"]+sys.argv[1:])
    path = args.path
    if os.path.isdir(path):
        print(cmdline)
        names = filter(lambda name:name.endswith(".ptx"),os.listdir(path)) 
        if args.namefilter is not None:
            names = filter(lambda name:name.find(args.namefilter) > -1, names)
        pass
        paths = map(lambda name:os.path.join(path, name), names)   
    else:
        paths = [path]
        args.all = True
    pass

    ptxs = map(PTX, paths)
    if not args.all:
        ptxs = filter( lambda ptx:ptx.d["TOTAL"] > 0, ptxs ) 
    pass
    print("\n".join(map(repr,ptxs)))



       
