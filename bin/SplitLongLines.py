#!/usr/bin/env python
"""
::

    cat /tmp/thrap-expandTest-i | CMakeOutput.py 

"""
import sys, re, os, logging, argparse
log = logging.getLogger(__name__)

class Line(object):
    long_cut = 150 
    def __init__(self, idx, txt):
        self.idx = idx
        self.txt = txt
        self.is_hdr = len(txt) > 2 and txt[0:2] == "--"
        self.is_long = len(txt) > self.long_cut 
        self.ltxt = txt.replace(" "," \\\n")

    hdr = property(lambda self:"[%3d;%3d]" % ( self.idx, len(self.txt) ))

    def __str__(self):
        return self.ltxt if self.is_long else self.txt
    def __repr__(self):
        return "%s %s " % ( self.hdr, self.txt) 


class Lines(object):
    def __init__(self, rawlines):
        self.lines = []
        for idx,txt in enumerate(rawlines):
            l = Line(idx+1, txt)
            if l.is_long:
                self.lines.append(Line(0,"##(\n"))
            pass
            self.lines.append(l)
            if l.is_long:
                self.lines.append(Line(0,"\n##)"))
            pass
        pass
    def __str__(self):
        return "\n".join(map(str, self.lines))


if __name__ == '__main__':

    rawlines = map(str.rstrip, sys.stdin.readlines() ) 
       
    ll = Lines(rawlines)
    print str(ll)


 

