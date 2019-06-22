#!/usr/bin/env python
"""
bouncelog.py
==============

Parse the kernel print log::

     tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog -DD   

          ## write kernel pindexlog for photon 1230

    boucelog.py 1230

          ## parse the log 


"""
from __future__ import print_function
from collections import OrderedDict
import os, sys, re


class Bounce(list):
    def __init__(self):
        list.__init__(self)
    def __str__(self):
       return "\n".join([""]+self+[""])


class BounceLog(OrderedDict):
    @classmethod
    def printlogpath(cls, pindex):
        return os.path.expandvars("$TMP/ox_%s.log" % pindex )

    BOUNCE = re.compile("bounce:(\S*)")

    def __init__(self, pindex):
        OrderedDict.__init__(self)
        self.pindex = pindex
        self.path = self.printlogpath(pindex)
        self.parse(self.path)
        
    def parse(self, path):
        self.lines = map(lambda line:line.rstrip(),file(path).readlines())

        curr = []
        bounce = -1

        for i, line in enumerate(self.lines):
            m = self.BOUNCE.search(line)
            if m:
                #bounce = int(m.group(1))   ## some OptiX rtPrintf bug makes bounce always 0 
                bounce += 1
                self[bounce] = Bounce()
            pass
            if bounce > -1:
                self[bounce].append(line)
            pass
            #print(" %3d : %3d : %s " % ( i, bounce,  line ))


if __name__ == '__main__':


    pindex = int(sys.argv[1]) if len(sys.argv) > 1 else 1230

    bl = BounceLog(pindex)

    for k, v in bl.items():
        print(k)
        print(v)
        print("\n\n") 

   

