#!/usr/bin/env python
"""
findpath.py
============

List the last 20 modified files with extension rst (EXT) within or below the invoking directory.

Configured via envars:

* EXT, default "rst"
* LAST, default "20" 

"""

import numpy as np, os, glob, time 

ext = os.environ.get("EXT", "rst")
last = int(os.environ.get("LAST", "20"))
print("\n--------- findpath.py ---------  EXT %s LAST %s " % (ext, last) )

ptn = '**/*.%s' % ext  

paths = list(glob.glob(ptn, recursive=True))
times = list(map(lambda p:os.path.getmtime(p), paths ))

order = np.argsort(times)[::-1][:last]     

for i in order:
    p = paths[i]
    t = time.gmtime(times[i])
    tstr = time.strftime("%c", t )
    print("%20s : %s " % (tstr, p))
pass

