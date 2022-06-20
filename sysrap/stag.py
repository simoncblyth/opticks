#!/usr/bin/env python 

import os, re
from collections import OrderedDict as odict 

class stag(object):
    lptn = re.compile("^\s*(\w+)\s*=\s*(.*?),*\s*?$")
    PATH = "$OPTICKS_PREFIX/include/sysrap/stag.h" 
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
   

         
