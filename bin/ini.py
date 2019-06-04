#!/usr/bin/env python

import sys
from opticks.ana.base import ini_


if __name__ == '__main__':
    ini = ini_(sys.argv[1])
    for k,v in sorted(ini.items(), reverse=True, key=lambda kv:float(kv[1])):
        print(" %50s : %10.4f " % (k,float(v)))
    pass
 
