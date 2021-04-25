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

    def imm(self, emm):
        return list(map(int, self.PTN.findall(emm))) 

    def label(self, emm):
        imm = self.imm(emm)
        labs = [self.mm[i] for i in imm] 
        lab = " ".join(labs)

        tilde = emm[0] == "t" or emm[0] == "~"
        pfx = (  "NOT: " if tilde else "     " ) 

        if emm == "~0":
            return "ALL"
        elif "," in emm:
            return ( "EXCL: " if tilde else "ONLY: " ) +  lab 
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
        self.sc = None   
        self.idx = None

    over_fast = property(lambda self:float(self.av)/float(self.sc.fast.av))
    over_slow = property(lambda self:float(self.av)/float(self.sc.slow.av))
    label = property(lambda self:self.sc.mm.label(self.emm))
    imm = property(lambda self:self.sc.mm.imm(self.emm))

    COLS = 6 
    def row(self):
        return  (int(self.idx), self.emm, self.av, self.over_fast, self.over_slow, self.label )

    def __repr__(self):
        return "[%2d] %4s %10.4f : %10.4f %10.4f : %s " % self.row()


class SnapScan(object):
    BASE = os.path.expandvars("$TMP/snap")
    def __init__(self, reldir, mm=None):
        self.mm = mm
        snapdir = os.path.join(self.BASE, reldir)
        paths = glob.glob("%s/*.json" % snapdir) 

        snaps = list(map(Snap,paths))
        snaps = sorted(snaps, key=lambda s:s.av)

        for s in snaps:
            s.sc = self
        pass

        # filter out the double emm
        snaps = list(filter(lambda s:len(s.imm) == 1, snaps))

        for idx, s in enumerate(snaps):
            s.idx = idx
        pass

        self.snaps = snaps

    def table(self):
        table = np.empty([len(self.snaps),Snap.COLS], dtype=np.object ) 
        for idx, snap in enumerate(self.snaps):
            table[idx] = snap.row()
        pass
        return table

    fast = property(lambda self:self.snaps[0])
    slow = property(lambda self:self.snaps[-1])

    def __repr__(self):
        return "\n".join(list(map(repr,self.snaps)))

    def jpg(self):
        return "\n".join(list(map(lambda s:s.jpg,self.snaps)))


class RSTTable(object):
     """
     Base class relying on subclasses to define labels and formats
     appropriate for the input table  
     """
     def divider(self, widths, char="-"):
         """
         +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
         """
         return "+".join([""]+list(map(lambda j:char*widths[j], range(len(widths))))+[""]) 

     def __init__(self, t):
         self.t = t  

     def __str__(self):

         nrow = self.t.shape[0]
         ncol = self.t.shape[1]    

         assert len(self.hfmt) == ncol
         assert len(self.rfmt) == ncol
         assert len(self.labels) == ncol
         assert len(self.wids) == ncol
         
         hfmt = "|".join( [""]+self.hfmt+[""])
         rfmt = "|".join( [""]+self.rfmt+[""])

         lines = []
         lines.append(self.divider(self.wids, "-")) 
         lines.append(hfmt % tuple(self.labels))
         lines.append(self.divider(self.wids, "="))
         for i in range(nrow):
             lines.append(rfmt % tuple(t[i]))
             lines.append(self.divider(self.wids,"-"))   
         pass
         return "\n".join(lines)    


class SnapTable(RSTTable):
    labels = ["idx", "emm", "time", "vslow", "vfast", "emm desc" ] 
    wids = [ 3, 10, 10, 10, 10, 70 ] 
    hfmt = [  "%3s", "%10s", "%10s",   "%10s",   "%10s",   "%-70s" ]  
    rfmt = [  "%3d", "%10s", "%10.4f", "%10.4f", "%10.4f", "%-70s" ]  

        

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

    t = ss.table()

    st = SnapTable(t)
    print(st)


