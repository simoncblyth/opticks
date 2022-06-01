#!/usr/bin/env python
import os, re,  numpy as np, logging, datetime
log = logging.getLogger(__name__)

from opticks.ana.key import keydir






class MM(object):
    """
    The input file is now a standard resident of the CSGFoundry directory 
    Note that the input file was formerly created using::

       ggeo.py --mmtrim > $TMP/mm.txt

    """
    PTN = re.compile("\d+")
    def __init__(self, path ):
        mm = os.path.expandvars(path)
        mm = open(mm, "r").read().splitlines() if os.path.exists(mm) else None
        self.mm = mm
        if mm is None:
            log.fatal("missing %s, which is now a standard part of CSGFoundry " % path  )
            sys.exit(1)
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


class LV(object):
    PTN = re.compile("\d+") 
    def __init__(self, path):
        lv = os.path.expandvars(path)
        lv = open(lv, "r").read().splitlines() if os.path.exists(lv) else None
        self.lv = lv
        if lv is None:
            log.fatal("missing %s, which is now a standard part of CSGFoundry " % path  )   
            sys.exit(1)
        pass

    def ilv(self, elv):
        return list(map(int, self.PTN.findall(elv))) 

    def label(self, elv):
        ilv = self.ilv(elv)
        mns = [self.lv[i] for i in ilv] 
        mn = " ".join(mns)
        tilde = elv[0] == "t" 
        lab = ""
        if elv == "t":
            lab = "ALL"
        else: 
            lab = ( "EXCL: " if tilde else "ONLY: " ) + mn
        pass
        return lab 

    def __str__(self):
        return "\n".join(self.lv)

    def __repr__(self):
        return "\n".join(["%3d:%s " % (i, self.lv[i]) for i in range(len(self.lv))]) 

    

class NPFold(object):
    INDEX = "NPFold_index.txt"

    @classmethod
    def IndexPath(cls, fold):
        return os.path.join(fold, cls.INDEX)

    @classmethod
    def HasIndex(cls, fold):
        return os.path.exists(cls.IndexPath(fold))

    @classmethod
    def namelist_to_namedict(cls, namelist):
        nd = {} 
        if not namelist is None:
            nd = dict(zip(range(len(namelist)),list(map(str,namelist)))) 
        pass
        return nd

    def __init__(self, fold):
        self.fold = fold
        if self.HasIndex(fold):
            self.load_idx(fold)
        else:
            self.load_fts(fold)
        pass

    def load_fts(self, fold):
        assert 0 

    def load_idx(self, fold):
        kk = open(self.IndexPath(fold),"r").read().splitlines()
        aa = []
        for k in kk:
            path = os.path.join(fold, k)
            assert os.path.exists(path)
            a = np.load(path)
            aa.append(a)
        pass
        self.kk = kk 
        self.aa = aa

    def find(self, k):
        return self.kk.index(k) if k in self.kk else -1 

    def has_key(self, k):
        return self.find(k) > -1  

    def __repr__(self):
        lines = []
        lines.append("NPFold %s " % self.fold )
        for i in range(len(self.kk)):
            k = self.kk[i]
            a = self.aa[i] 
            stem, ext = os.path.splitext(os.path.basename(k))
            path = os.path.join(self.fold, k)  
            lines.append("%10s : %20s : %s " % (stem, str(a.shape), k ))
        pass
        return "\n".join(lines) 
        

class SSim(NPFold):
    BND = "bnd.npy"
    @classmethod
    def Load(cls, simbase):
        sim = cls(fold=os.path.join(simbase, "SSim"))  
        return sim

    def __init__(self, fold):
        NPFold.__init__(self, fold) 
        if self.has_key(self.BND):
            bnpath = os.path.join(fold, "bnd_names.txt")
            assert os.path.exists(bnpath)
            bnd_names = open(bnpath,"r").read().splitlines()
            self.bnd_names = bnd_names
        pass
        if hasattr(self, 'bnd_names'):  # names list from NP bnd.names metadata 
             bndnamedict = NPFold.namelist_to_namedict(self.bnd_names)
        else:
             bndnamedict = {}
        pass
        self.bndnamedict = bndnamedict
        pass



class CSGFoundry(object):
    FOLD = os.path.expandvars("$TMP/CSG_GGeo/CSGFoundry")
    FMT = "   %10s : %20s  : %s "

    @classmethod
    def CFBase(cls):
        """
        Precedence order for which envvar is used to derive the CFBase folder
        This allows scripts such as CSGOptiX/cxsim.sh to set envvars to control 
        the geometry arrays loaded. For example this is used such that local 
        analysis on laptop can use the geometry rsynced from remote.  

        1. CFBASE_LOCAL
        2. CFBASE
        3. GEOM
        4. OPTICKS_KEY

        """
        if "CFBASE_LOCAL" in os.environ:
            cfbase= os.environ["CFBASE_LOCAL"]
        elif "CFBASE" in os.environ:
            cfbase= os.environ["CFBASE"]
        elif "GEOM" in os.environ:
            geom = os.environ["GEOM"]
            rel = "GeoChain_Darwin" 
            cfbase = os.path.expandvars("/tmp/$USER/opticks/%s/%s" %(rel, geom) )  # guess
        elif "OPTICKS_KEY" in os.environ:
            kd = keydir(os.environ["OPTICKS_KEY"])
            cfbase = os.path.join(kd, "CSG_GGeo")
            ## HMM: match against OPTICKS_KEY_* in environ to find the "tag" of this key 
        else:
            cfbase = None
        pass
        return cfbase

    @classmethod
    def Load(cls):
        cfbase = cls.CFBase()
        print("cfbase:%s " % cfbase)
        assert not cfbase is None
        cf = cls(fold=os.path.join(cfbase, "CSGFoundry"))
        return cf     

    @classmethod
    def FindDirUpTree(cls, origpath, name="CSGFoundry"): 
        """
        :param origpath: to traverse upwards looking for sibling dirs with *name*
        :param name: sibling directory name to look for 
        :return full path to *name* directory 
        """
        elem = origpath.split("/")
        found = None
        for i in range(len(elem),0,-1):
            path = "/".join(elem[:i])
            cand = os.path.join(path, name)
            log.debug(cand) 
            if os.path.isdir(cand):
                found = cand
                break 
            pass  
        pass
        return found 

    @classmethod
    def FindDigest(cls, path):
        if path.find("*") > -1:  # eg with a glob pattern filename element
            base = os.path.dirname(path)
        else:
            base = path
        pass 
        base = os.path.expandvars(base)
        return cls.FindDigest_(base)

    @classmethod
    def FindDigest_(cls, path):
        """
        :param path: Directory path in which to look for a 32 character digest path element 
        :return 32 character digest or None: 
        """
        hexchar = "0123456789abcdef" 
        digest = None
        for elem in path.split("/"):
            if len(elem) == 32 and set(elem).issubset(hexchar):
                digest = elem
            pass
        pass
        return digest 

    #def __init__(self, fold=FOLD):
    def __init__(self, fold):
        self.load(fold)
        self.meshnamedict = NPFold.namelist_to_namedict(self.meshname)
        self.primIdx_meshname_dict = self.make_primIdx_meshname_dict()  

        self.sim = SSim.Load(fold)

        self.mokname = "zero one two three four five six seven eight nine".split()
        self.moknamedict = NPFold.namelist_to_namedict(self.mokname)
        self.insnamedict = {}

        self.lv = LV(os.path.join(fold, "meshname.txt"))
        self.mm = MM(os.path.join(fold, "mmlabel.txt"))


    def meshIdx(self, primIdx):
        """
        """
        assert primIdx < len(self.prim)
        midx = self.prim[primIdx].view(np.uint32)[1,1]
        return midx 

    def make_primIdx_meshname_dict(self):
        """
        See notes/issues/cxs_2d_plotting_labels_suggest_meshname_order_inconsistency.rst
        this method resolved an early naming bug 

        CSG/CSGPrim.h:: 

             95     PRIM_METHOD unsigned  meshIdx() const {           return q1.u.y ; }  // aka lvIdx
             96     PRIM_METHOD void   setMeshIdx(unsigned midx){     q1.u.y = midx ; }

        """
        d = {}
        for primIdx in range(len(self.prim)):
            midx = self.meshIdx (primIdx)      # meshIdx method with contiguous primIdx argument
            assert midx < len(self.meshname)
            mnam = self.meshname[midx]
            d[primIdx] = mnam
            #print("CSGFoundry:primIdx_meshname_dict primIdx %5d midx %5d meshname %s " % (primIdx, midx, mnam))
        pass
        return d

    def loadtxt(self, path):
        """
        When the path contains only a single line using dtype np.object or |S100
        both result in an object with no length::
        
            array('solidXJfixture', dtype=object)

        """
        # both these do not yield an array when only a single line in the file
        # 
        #a_txt = np.loadtxt(path, dtype=np.object)   # yields single object 
        #a_txt = np.loadtxt(path, dtype="|S100")

        txt = open(path).read().splitlines()
        a_txt = np.array(txt, dtype=np.object ) 
        return a_txt 
  

    def load(self, fold):
        log.info("load %s " % fold)

        if not os.path.isdir(fold):
            log.fatal("CSGFoundry folder %s does not exist " % fold)
            log.fatal("create foundry folder from OPTICKS_KEY geocache with CSG_GGeo/run.sh " )
            assert 0 
        pass

        names = os.listdir(fold)
        stems = []
        stamps = []
        for name in filter(lambda name:name.endswith(".npy") or name.endswith(".txt"), names):
            path = os.path.join(fold, name)
            st = os.stat(path)
            stamp = datetime.datetime.fromtimestamp(st.st_ctime)
            stamps.append(stamp)
            stem = name[:-4]
            a = np.load(path) if name.endswith(".npy") else self.loadtxt(path)
            setattr(self, stem, a)
            stems.append(stem)
        pass

        min_stamp = min(stamps)
        max_stamp = max(stamps)
        now_stamp = datetime.datetime.now()
        dif_stamp = max_stamp - min_stamp
        age_stamp = now_stamp - max_stamp 

        #print("min_stamp:%s" % min_stamp)
        #print("max_stamp:%s" % max_stamp)
        #print("dif_stamp:%s" % dif_stamp)
        #print("age_stamp:%s" % age_stamp)

        assert dif_stamp.microseconds < 1e6, "stamp divergence detected microseconds %d : so are seeing mixed up results from multiple runs " % dif_stamp.microseconds

        self.stems = stems
        self.min_stamp = min_stamp
        self.max_stamp = max_stamp
        self.age_stamp = age_stamp
        self.stamps = stamps
        self.fold = fold

    def desc(self, stem):
        a = getattr(self, stem)
        ext = ".txt" if a.dtype == 'O' else ".npy"
        pstem = "bnd" if stem == "bndname" else stem 
        path = os.path.join(self.fold, "%s%s" % (pstem, ext))
        return self.FMT % (stem, str(a.shape), path) 

    def head(self):
        return "\n".join(map(str, [self.fold, "min_stamp:%s" % self.min_stamp, "max_stamp:%s" % self.max_stamp, "age_stamp:%s" % self.age_stamp])) 

    def body(self):
        return "\n".join(map(lambda stem:self.desc(stem),self.stems))
 
    def __repr__(self):
        return "\n".join([self.head(),self.body()])

    def dump_node_boundary(self):
        logging.info("dump_node_boundary") 
        node = self.node

        node_boundary = node.view(np.uint32)[:,1,2]
        ubs, ubs_count = np.unique(node_boundary, return_counts=True)

        bndname = getattr(self, 'bndname', None)

        for ub, ub_count in zip(ubs, ubs_count):
            bn = bndname[ub] if not bndname is None else "-"
            print(" %4d : %6d : %s " % (ub, ub_count, bn))
        pass 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    cf = CSGFoundry.Load()
    print(cf)
    
    #cf.dump_node_boundary()
    #d = cf.primIdx_meshname_dict()
    #print(d)    


