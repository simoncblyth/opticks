#!/usr/bin/env python
"""
scanvers.py
==============

Present the scan notes for versions provided in integer arguments, 
or present all.

Usage::

    scanvers.py 
    scanvers.py 8 9

"""
import sys, re
from collections import OrderedDict as odict
from opticks.ana.base import _subprocess_output

class ScanVersNotes(object):
    """
    Parse the output of bashfunc that holds notes about 
    the intent of each scan version. 
    """
    bashfunc = "scan-;scan-vers-notes"  
    startptn = re.compile("^(\d*)\s*$")  # integer starting a blank line
    blankptn = re.compile("^\s*$")   # blank line
    def __init__(self):
        out,err = _subprocess_output(["bash","-lc",self.bashfunc])
        assert err == ""
        self.d = self.parse(out) 

    def parse(self, out):
        d = odict() 
        v = None
        for line in out.split("\n"):
            mitem = self.startptn.match(line)
            mend = self.blankptn.match(line)
            if mend:break   
            pass 
            if mitem:
                 v = int(mitem.groups()[0]) 
                 d[v] = []
            else:
                 if not v is None: 
                     d[v].append(line)  
            pass
            pass
        return d
       
    def item(self, v):
        return "%s\n%s" % (v,"\n".join(self.d[v]))

    def __call__(self, v):
        return self.item(v)
 
    def __str__(self):
        return "\n".join([self.item(v) for v in self.d.keys()])


if __name__ == '__main__':
    vn = ScanVersNotes()
    if len(sys.argv) == 1:
        print(vn) 
    else:
        for v in map(int, sys.argv[1:]):
            print(vn(v))
        pass
    pass


