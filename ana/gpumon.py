#!/usr/bin/env python

import os, sys, numpy as np
from collections import OrderedDict as odict 

decode_ = lambda v:"".join(map( lambda c:str(unichr(c)), filter(None,map(lambda i:(int(v) >> i*8) & 0xff, range(8))) ))


class NGPU(object):
    def __init__(self, path):
        a = np.load(os.path.expandvars(path))
        d = odict()
        tot_bytes = 0
        for i in range(len(a)):
            name, owner, note = map(decode_, a[i,:3])
            loc = "/".join([name, owner, note])
            num_bytes = a[i,3]
            d[loc] = num_bytes 
            tot_bytes += num_bytes
        pass
        self.d = d 
        self.a = a 
        self.tot_bytes = tot_bytes
        self.path = path 

    def __str__(self):
        return "\n".join([" %30s : %s " % (k,self.brief(k)) for k, v in self.d.items()]) 
    def __repr__(self):
        return "%10d : %15d : %6.2f : %s " % ( len(self.d), self.tot_bytes, float(self.tot_bytes)/1e6, self.path  )  

    def brief(self, loc):
        nb = self.d.get(loc, -1)
        return " %15d : %6.2f  " % ( nb, float(nb)/1e6 ) if nb > -1 else ""

    def index(self, loc):
        return self.d.keys().index(loc) if self.d.has_key(loc) else -1 



if __name__ == '__main__':

     #a_name = "OKTest"
     a_name = "OTracerTest"
     b_name = "OKX4Test"
 
     a_path = sys.argv[1] if len(sys.argv) > 1 else "$TMP/%s_GPUMon.npy" % a_name
     b_path = sys.argv[2] if len(sys.argv) > 2 else "$TMP/%s_GPUMon.npy" % b_name
     
     a = NGPU(a_path)
     b = NGPU(b_path)

     print repr(a)
     print repr(b)

     locs = set(a.d.keys()).union(set(b.d.keys()))

     fmt = " %30s :  %30s  :  %30s   "
     print fmt % ( len(locs) , a_path , b_path )  
     for loc in sorted(locs, key=lambda loc:max(a.index(loc), b.index(loc))):
         print fmt % ( loc, a.brief(loc),  b.brief(loc) ) 
     pass




