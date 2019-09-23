#!/usr/bin/env python
"""
bashnotes.py
==============

Present the notes from bash/RST definition lists with integer keys, 
presenting individual notes or all.

Usage::

    bashnotes.py 
    bashnotes.py 8 9
    bashnotes.py 8 9 --bashcmd "scan-;scan-ph-notes"

"""
import sys, re, argparse
from collections import OrderedDict as odict
from opticks.ana.base import _subprocess_output

class BashNotes(object):
    """
    Parse the output of bashfunc that is assumed to return 
    an RST style definition list keyed with integers. 
    """
    startptn = re.compile("^(\d*)\s*$")  # integer starting a blank line
    blankptn = re.compile("^\s*$")   # blank line
    def __init__(self, bashcmd):
        out,err = _subprocess_output(["bash","-lc",bashcmd])
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

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument( "--bashcmd", default="scan-;scan-ph-notes", help="Bash command returning a definition list keyed with integers" )
    parser.add_argument( "vers", nargs="*", default=[1], type=int, help="Prefix beneath which to search for OpticksProfile.npy" )
    args = parser.parse_args()
 
    bn = BashNotes(args.bashcmd)
    if len(args.vers) == 0:
        print(bn) 
    else:
        for v in args.vers:
            print(bn(v))
        pass
    pass


