#!/usr/bin/env python
"""
::

    ggeo.py --mm > $TMP/mm.txt    # create list of mm names used for labels

    snap.py       # list the snaps in speed order with labels 

    open $(snap.py --jpg)         # open the jpg ordered by render speed


"""
import os, logging, glob, json, re, argparse 
import numpy as np

class MM(object):
    PTN = re.compile("\d+") 
    def __init__(self, path="$TMP/mm.txt"):
        mm = os.path.expandvars(path)
        mm = open(mm, "r").read().splitlines() if os.path.exists(mm) else None
        self.mm = mm

    def label(self, emm):
        m = self.PTN.search(emm)
        imm = int(m.group())
        assert imm < len(self.mm) 
        lab = " %s" % self.mm[imm]

        tilde = emm[0] == "t" or emm[0] == "~"
        pfx = (  "NOT" if tilde else "   " ) 

        if emm == "~0":
            return "ALL"
        elif "," in emm:
            return ( "EXCL" if tilde else "ONLY" ) +  lab 
        else:
            return lab  
        pass

    def __repr__(self):
        return "\n".join(self.mm)


class Snap(object):

    def __init__(self, path):
        path = path 
        js = json.load(open(path,"r"))
        jpg = path.replace(".json","00000.jpg")

        emm = js['emm']
        self.av = js['av'] 
        self.emm = emm
        self.path = path 
        self.js = js 
        self.jpg = jpg 
        # below are set by SnapScan after sorting
        self.fast = None   
        self.slow = None
        self.mm = None   

    over_fast = property(lambda self:float(self.av)/float(self.fast.av))
    over_slow = property(lambda self:float(self.av)/float(self.slow.av))
    label = property(lambda self:self.mm.label(self.emm))

    def __repr__(self):
        return "Snap %4s %10.4f : %10.4f %10.4f : %s " % (self.emm, self.av, self.over_fast, self.over_slow, self.label )


class SnapScan(object):
    BASE = os.path.expandvars("$TMP/snap")
    def __init__(self, reldir, mm=None):
        snapdir = os.path.join(self.BASE, reldir)
        paths = glob.glob("%s/*.json" % snapdir) 
        snaps = list(map(Snap,paths))
        snaps = sorted(snaps, key=lambda s:s.av)

        for s in snaps:
            s.fast = snaps[0] 
            s.slow = snaps[-1]
            s.mm = mm 
        pass
        self.snaps = snaps
        self.mm = mm

    def __repr__(self):
        return "\n".join(list(map(repr,self.snaps)))

    def jpg(self):
        return "\n".join(list(map(lambda s:s.jpg,self.snaps)))
  



def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    parser.add_argument(  "--reldir", default="lLowerChimney_phys", help="Relative dir beneath $TMP/snap from which to load snap .json" ) 
    parser.add_argument(  "--jpg", action="store_true", help="List jpg paths in speed order" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args  


if __name__ == '__main__':

    args = parse_args(__doc__)
    mm = MM("$TMP/mm.txt")

    ss = SnapScan(args.reldir, mm) 
    if args.jpg:
        print(ss.jpg())
    else:
        print(ss) 
    pass

