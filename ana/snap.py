#!/usr/bin/env python
"""
::

    ggeo.py --mmtrim    # create list of mm names used for labels at $TMP/mm.txt

    snap.py       # list the snaps in speed order with labels 

    open $(snap.py --jpg)         # open the jpg ordered by render speed


"""
import os, logging, glob, json, re, argparse 
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.rsttable import RSTTable


class MM(object):
    """
    TODO: use CSGFoundry/mmlabel.txt instead of $TMP/mm.txt
    """
    PTN = re.compile("\d+") 
    def __init__(self, path="$TMP/mm.txt"):
        mm = os.path.expandvars(path)
        mm = open(mm, "r").read().splitlines() if os.path.exists(mm) else None
        self.mm = mm
        if mm is None:
            print("missing mm.txt/mmtrim.txt, create with : ggeo.py --mmtrim > $TMP/mm.txt  " ) 
        pass

    def imm(self, emm):
        return list(map(int, self.PTN.findall(emm))) 

    def label(self, emm):
        imm = self.imm(emm)
        labs = [self.mm[i] for i in imm] 
        lab = " ".join(labs)

        tilde = emm[0] == "t" or emm[0] == "~"
        pfx = (  "NOT: " if tilde else "     " ) 

        if emm == "~0" or emm == "t0":
            return "ALL"
        elif imm == [1,2,3,4]:
            return "ONLY PMT"
        elif "," in emm:
            return ( "EXCL: " if tilde else "ONLY: " ) +  lab 
        else:
            return lab  
        pass

    def __repr__(self):
        return "\n".join(self.mm)


class Snap(object):
    @classmethod
    def is_valid(cls, jpg_path):
        #json_path = cls.find_json(jpg_path) 
        json_path = jpg_path.replace(".jpg", ".json")
        valid = (not json_path is None) and json_path[-5:] == ".json"
        log.debug("is_valid %r %r %r " % (jpg_path, json_path, valid ) ) 
        return valid 

    def __init__(self, jpg_path):
        json_path = jpg_path.replace(".jpg", ".json")
        log.debug("jpg_path %s json_path %s " % (jpg_path, json_path))          
        js = json.load(open(json_path,"r"))

        self.js = js 
        self.jpg = jpg_path
        self.path = json_path 
        self.av = js['av'] 
        self.emm = js['emm']
        self.argline = js['argline']

        # below are set by SnapScan after sorting
        self.sc = None   
        self.idx = None

    def jpg_(self):
        """
        tilde ~ and __t0__ causing RST problems : so replace tham 
        """
        fold = os.path.dirname(self.jpg)
        tname = self.jpg_tname()
        return os.path.join(fold, tname)

    def jpg_tname(self): 
        """
        This is a workaround for problems with s5 machinery for names 
        containing strings like the below, presumably due to some RST meaning 
        the names get mangled preventing the association between presentation pages and 
        background image definition causes the images to not appear::

            __~0__
            __t0__
            _ALL_

        ACTUALLY THIS ISSUE MAY BE FROM NON-UNIQUE IDENTIFIERS DERIVED FROM THE 
        SLIDE TITLES BY REMOVING SOME CHARS SUCH AS "," "_" "~"
        """
        name = os.path.basename(self.jpg)
        tname = name.replace("~","t")
        tname = tname.replace("__t0__", "_all_") 
        return tname 

    def mvjpg(self):
        name = os.path.basename(self.jpg)
        tname = self.jpg_tname()
        return None if name == tname else "mv %s %s" % (name, tname)

    def cpjpg(self, pfx, s5base):
        s5base = os.path.expandvars(s5base)
        name = os.path.basename(self.jpg)
        ppath = "%s%s/%s"%(s5base,pfx,name)
        if os.path.exists(ppath):
            ret = None   
        else:
            ret = "cp %s %s" % (self.jpg, ppath)
        return ret 
 

    def refjpg(self, pfx, afx="1280px_720px", indent="    "): 
        """
        For inclusion into s5_background_image.txt 
        """
        name = os.path.basename(self.jpg_())
        return "\n".join([indent+self.title(), indent+pfx+"/"+name+" "+afx, ""])

    def title(self):
        name = os.path.basename(self.jpg_())
        stem = name.replace(".jpg","") 
        return "[%d]%s" % (self.idx, stem)
 
    def pagetitle(self,kls="blue"):
        return ":%s:`%s`" % (kls,self.title())
 
    def pagejpg(self):
        title = self.pagetitle()
        return "\n".join([title, "-" * len(title), ""])


    over_fast = property(lambda self:float(self.av)/float(self.sc.fast.av))
    over_slow = property(lambda self:float(self.av)/float(self.sc.slow.av))
    over_candle = property(lambda self:float(self.av)/float(self.sc.candle.av))

    label = property(lambda self:self.sc.mm.label(self.emm))
    imm = property(lambda self:self.sc.mm.imm(self.emm))

    def row(self):
        return  (int(self.idx), self.emm, self.av, self.over_candle, self.label )


    SPACER = "    "
    LABELS = ["idx", "-e", "time(s)", "relative", "enabled geometry description" ] 
    WIDS = [     3,     10,     10,        10,       70 ] 
    HFMT = [  "%3s",  "%10s", "%10s",   "%10s",   "%-70s" ]  
    RFMT = [  "%3d", "%10s", "%10.4f", "%10.4f",  "%-70s" ]  
    PRE  =  [  ""   , ""    , SPACER , SPACER  ,  SPACER ] 
    POST =  [  ""   , ""    , SPACER , SPACER  ,  SPACER ]  


    def __repr__(self):
        RFMT = [ self.PRE[i] + self.RFMT[i] +self.POST[i] for i in range(len(self.RFMT))]
        rfmt = " ".join(RFMT)
        #print(rfmt)
        return rfmt % self.row()

    @classmethod
    def Hdr(cls):
        HFMT = [cls.PRE[i] + cls.HFMT[i] + cls.POST[i] for i in range(len(cls.HFMT))]
        hfmt = " ".join(HFMT)
        return hfmt % tuple(cls.LABELS)




class SnapScan(object):

    @classmethod
    def MakeSnaps(cls, globptn):
        log.info("globptn %s " % globptn )
        raw_paths = glob.glob(globptn) 
        log.debug("raw_paths %d : 1st %s " % (len(raw_paths), raw_paths[0]))
        paths = filter(lambda p:Snap.is_valid(p), raw_paths)
        snaps = list(map(Snap,paths))
        snaps = sorted(snaps, key=lambda s:s.av)
        return snaps

    @classmethod
    def SelectSnaps(cls, all_snaps ):
        snaps = []
        for s in all_snaps:  
            #print(s.imm)
            if len(s.imm) == 1 or s.imm == [8, 0] or s.imm == [1,2,3,4]:
                snaps.append(s)
            else:
                log.debug("skip %s " % str(s.imm))
            pass   
        pass
        return snaps

    def __init__(self, globptn, mm=None, candle_emm="1,2,3,4"):
        """
        :param globptn: eg /tmp/blyth/opticks/CSGOptiX/CSGOptiXRender/CSG_GGeo/cvd1/70000/cxr_overview/cam_0_emm_*/*.jpg
        """
        self.mm = mm
        all_snaps = self.MakeSnaps(globptn)
        candle = None
        for s in all_snaps:
            s.sc = self
            if s.emm == candle_emm:
                candle = s
            pass  
        pass

        # filter out the double emm
        #snaps = list(filter(lambda s:len(s.imm) != 2, snaps))
        snaps = self.SelectSnaps(all_snaps)

        for idx, s in enumerate(snaps):
            s.idx = idx
        pass
        self.snaps = snaps
        self.candle = candle 



    def table(self):
        table = np.empty([len(self.snaps),len(Snap.LABELS)], dtype=np.object ) 
        for idx, snap in enumerate(self.snaps):
            table[idx] = snap.row()
        pass
        return table

    fast = property(lambda self:self.snaps[0])
    slow = property(lambda self:self.snaps[-1])

    def __repr__(self):
        return "\n".join( [Snap.Hdr()] + list(map(repr,self.snaps)) + [Snap.Hdr()] )

    def jpg(self):
        return "\n".join(list(map(lambda s:s.jpg,self.snaps)))

    def mvjpg(self):
        return "\n".join(list(filter(None,map(lambda s:s.mvjpg(),self.snaps))))

    def cpjpg(self, pfx, s5base):
        return "\n".join(list(filter(None,map(lambda s:s.cpjpg(pfx, s5base),self.snaps))))

    def argline(self):
        return "\n".join(list(map(lambda s:s.argline,self.snaps)))

    def refjpg(self, pfx): 
        return "\n".join(list(map(lambda s:s.refjpg(pfx),self.snaps)))

    def pagejpg(self): 
        return "\n".join(list(map(lambda s:s.pagejpg(),self.snaps)))

        

def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(  "--level", default="info", help="logging level" ) 
    parser.add_argument(  "--globptn", default="$TMP/snap/*.jpg", help="base" ) 
    parser.add_argument(  "--jpg", action="store_true", help="List jpg paths in speed order" ) 
    parser.add_argument(  "--refjpgpfx", default="/env/presentation/snap/lLowerChimney_phys", help="List jpg paths s5 background image presentation format" ) 
    parser.add_argument(  "--s5base", default="$HOME/simoncblyth.bitbucket.io", help="Presentation repo base" )
    parser.add_argument(  "--refjpg", action="store_true", help="List jpg paths s5 background image presentation format" ) 
    parser.add_argument(  "--pagejpg", action="store_true", help="List jpg for inclusion into s5 presentation" ) 
    parser.add_argument(  "--mvjpg", action="store_true", help="List jpg for inclusion into s5 presentation" ) 
    parser.add_argument(  "--cpjpg", action="store_true", help="List cp commands to place into presentation repo" ) 
    parser.add_argument(  "--argline", action="store_true", help="List argline in speed order" ) 
    parser.add_argument(  "--rst", action="store_true", help="Dump table in RST format" ) 
    parser.add_argument(  "--snaps", action="store_true", help="Debug: just create SnapScan " ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args  


if __name__ == '__main__':
    args = parse_args(__doc__)
    log.debug(" args %s " % str(args))
    mm = MM("$TMP/mm.txt")

    ss = SnapScan(args.globptn, mm) 
    if args.jpg:
        print(ss.jpg())
    elif args.refjpg:
        print(ss.refjpg(args.refjpgpfx))
    elif args.pagejpg:
        print(ss.pagejpg())
    elif args.mvjpg:
        print(ss.mvjpg())
    elif args.cpjpg:
        print(ss.cpjpg(args.refjpgpfx, args.s5base))
    elif args.argline:
        print(ss.argline())
    elif args.snaps:
        snaps = ss.snaps
    elif args.rst:
        t = ss.table()
        rst = RSTTable.Render(t, Snap.LABELS, Snap.WIDS, Snap.HFMT, Snap.RFMT, Snap.PRE, Snap.POST )
        print(rst)
    else:
        print(ss) 
    pass



