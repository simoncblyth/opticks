#!/usr/bin/env python
"""
NPMeta.py
===========

Parsing metadata lines from NP.hh 

TODO: introspective listing of keys, rather than current needing to know whats there 

"""

import os, logging
import numpy as np
log = logging.getLogger(__name__)

class NPMeta(object):

    ENCODING = "utf-8"

    @classmethod
    def AsDict(cls, meta):
        d = {}
        for item in meta:
            line = item.decode(cls.ENCODING)
            dpos = line.find(":") 
            if dpos > -1:
               key = line[:dpos]
               val = line[dpos+1:] 
               d[key] = val 
            pass
        pass    
        return d 

    def __init__(self, meta):
        self.meta = meta  
        self.d = self.AsDict(meta)
           
    def __len__(self):
        return len(self.meta)  

    def find(self, k_start, fallback=None, encoding=ENCODING):
        meta = self.meta
        ii = np.flatnonzero(np.char.startswith(meta, k_start.encode(encoding)))  
        log.debug( " ii %s len(ii) %d  " % (str(ii), len(ii)) )
        ret = fallback 
        if len(ii) == 1:
            i = ii[0]
            line = meta[i].decode(encoding)
            ret = line[len(k_start):]
            log.debug(" line [%s] ret [%s] " % (line,ret) )
        else:
            log.debug("did not find line starting with %s or found more than 1" % k_start) 
        pass
        return ret 

    def __str__(self):
        return "\n".join(map(str,self.meta))
    def __repr__(self):
        return repr(self.d)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    path = "/tmp/t.txt"
    meta = np.loadtxt(path, dtype="|S100", delimiter="\t" )
    pm = NPMeta(meta)

    moi = pm.find("moi:")
    midx = pm.find("midx:")
    mord = pm.find("mord:")
    iidx = pm.find("iidx:")
    print(" moi:[%s] midx:[%s] mord:[%s] iidx:[%s] " % (moi, midx, mord, iidx) )

    TOPLINE = pm.find("TOPLINE:")
    BOTLINE = pm.find("BOTLINE:")

    print(" TOPLINE:[%s] " % TOPLINE )
    print(" BOTLINE:[%s] " % BOTLINE )







    


