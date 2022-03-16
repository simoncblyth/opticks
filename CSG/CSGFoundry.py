#!/usr/bin/env python
import os, numpy as np, logging, datetime
log = logging.getLogger(__name__)

class CSGFoundry(object):
    FOLD = os.path.expandvars("$TMP/CSG_GGeo/CSGFoundry")
    FMT = "   %10s : %20s  : %s "

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


    @classmethod
    def namelist_to_namedict(cls, namelist):
        nd = {} 
        if not namelist is None:
            nd = dict(zip(range(len(namelist)),list(map(str,namelist)))) 
        pass
        return nd
 
    def __init__(self, fold=FOLD):
        self.load(fold)
        self.meshnamedict = self.namelist_to_namedict(self.meshname)

        if hasattr(self, 'bnd_meta'):
             bndnamedict = self.namelist_to_namedict(self.bnd_meta)
        else:
             bndnamedict = {}
        pass
        self.bndnamedict = bndnamedict

        self.mokname = "zero one two three four five six seven eight nine".split()
        self.moknamedict = self.namelist_to_namedict(self.mokname)
        self.insnamedict = {}

    def meshIdx(self, primIdx):
        """
        """
        assert primIdx < len(self.prim)
        midx = self.prim[primIdx].view(np.uint32)[1,1]
        return midx 

    def primIdx_meshname_dict(self):
        """
        See notes/issues/cxs_2d_plotting_labels_suggest_meshname_order_inconsistency.rst
        """
        d = {}
        for primIdx in range(len(self.prim)):
            midx = self.meshIdx (primIdx) 
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
            if name == "bnd.txt": stem = "bndname"  ## TODO: avoid clash of stems between bnd.npy and bnd.txt ?
            setattr(self, stem, a)
            stems.append(stem)
            #globals()[stem] = a 
            #print(self.FMT % (stem, str(a.shape), path))
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
        bndname = self.bndname

        node_boundary = node.view(np.uint32)[:,1,2]
        ubs, ubs_count = np.unique(node_boundary, return_counts=True)

        for ub, ub_count in zip(ubs, ubs_count):
            print(" %4d : %6d : %s " % (ub, ub_count, bndname[ub]))
        pass 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    geom = os.environ.get("GEOM", "body_phys")
    #rel = "GeoChain" 
    rel = "GeoChain_Darwin" 
    base = os.path.expandvars("/tmp/$USER/opticks/%s/%s" %(rel, geom) ) 
    cf = CSGFoundry(os.path.join(base, "CSGFoundry"))
    cf.dump_node_boundary()

    d = cf.primIdx_meshname_dict()
    print(d)    


