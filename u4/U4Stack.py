#!/usr/bin/env python

import numpy as np
import os, re, logging
log = logging.getLogger(__name__)
from collections import OrderedDict as odict 

class U4Stack_item(object):
    @classmethod
    def Placeholder(cls):
        return cls(-1,"placeholder","ERROR" )

    def __init__(self, code, name, note=""):
        self.code = code 
        self.name = name 
        self.note = note 

    def __repr__(self):
        return "%2d : %10s : %s " % (self.code, self.name, self.note)

class U4Stack(object):
    PATH = "$OPTICKS_PREFIX/include/u4/U4Stack.h" 
    enum_ptn = re.compile("^\s*(\w+)\s*=\s*(.*?),*\s*?$")

    def __init__(self, path=PATH):
        path = os.path.expandvars(path)
        lines = open(path, "r").read().splitlines()
        self.path = path
        self.lines = lines
        self.items = []
        self.d = self.parse()

    def parse(self):
        d=odict()
        for line in self.lines:
            enum_match = self.enum_ptn.match(line)
            if enum_match:
                name, val = enum_match.groups()
                pfx = "U4Stack_"
                assert name.startswith(pfx)
                sname = name[len(pfx):]
                code = int(val)

                item = U4Stack_item(code, sname, "")
                self.items.append(item)
                d[code] = item
                log.debug(" name %20s sname %10s val %5s code %2d " % (name, sname, val, code))     
            else:
                pass
                log.debug(" skip :  %s " % line )
            pass 
        pass
        return d

    def label(self, st):
        d = self.d
        label_ = lambda _:repr(d.get(_,U4Stack_item.Placeholder()))
        ilabel_ = lambda _:"%2d : %s" % ( _, label_(st[_])) 
        return "\n".join(map(ilabel_, range(len(st))))
 
    def __str__(self):
        return "\n".join(self.lines)

    def __repr__(self):
        return "\n".join(list(map(repr,self.items)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    stack = U4Stack()
    #print(stack) 
    print(repr(stack))
    
    st = np.array([[2, 6, 4, 3, 8, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2, 6, 4, 3, 8, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2, 6, 4, 3, 8, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    print(stack.label(st[0,:10]))
    



