#!/usr/bin/env python 

import numpy as np
import os, re
from collections import OrderedDict as odict 

class stag(object):
    """
    # the below NSEQ, BITS, ... param need to correspond to stag.h static constexpr 
    """
    lptn = re.compile("^\s*(\w+)\s*=\s*(.*?),*\s*?$")
    PATH = "$OPTICKS_PREFIX/include/sysrap/stag.h" 

    NSEQ = 2
    BITS = 5 
    MASK = ( 0x1 << BITS ) - 1 
    SLOTMAX = 64//BITS
    SLOTS = SLOTMAX*NSEQ

    @classmethod
    def Split(cls, tag):
        st = np.zeros( (len(tag), cls.SLOTS), dtype=np.uint8 )   
        for i in range(cls.NSEQ):
            for j in range(cls.SLOTMAX):
                st[:,i*cls.SLOTMAX+j] = (tag[:,i] >> (cls.BITS*j)) & cls.MASK
            pass
        pass
        return st 

    def __init__(self, path=PATH):
        path = os.path.expandvars(path)
        lines = open(path, "r").read().splitlines()
        self.path = path 
        self.lines = lines 
        self.d = self.parse()

    def parse(self):
        d=odict()
        for line in self.lines:
            m = self.lptn.match(line)
            if m:
                name, val = m.groups() 
                d[val] = name
                print("%40s : name:%20s val:%10s " % (line,name,val) )
            pass
        pass
        return d
 
    def __str__(self):
        return "\n".join(self.lines)

    def __repr__(self):
        return "\n".join(["%20s : %s" % (v, self.d[v] ) for v in self.d])  

if __name__ == '__main__':
   s = stag()
   print(s) 
   print(repr(s))
   

         
