#!/usr/bin/env python

import os, logging, numpy as np, datetime
log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

class Fold(object):
    @classmethod
    def Load(cls, *args, **kwa):
        relbase = os.path.join(*args[1:]) if len(args) > 1 else args[0]
        kwa["relbase"] = relbase   # relbase is the dir path excluding the first element 
        base = os.path.join(*args)
        base = os.path.expandvars(base) 
        quiet = kwa.get("quiet", False) == True 

        fold = cls(base, **kwa) if os.path.isdir(base) else None
        if fold is None and quiet == False:
            log.error("failed to load from base [%s]" % base )
        pass
        return fold

    def __init__(self, base, **kwa):
        self.base = base
        self.kwa = kwa 
        self.relbase = kwa.get("relbase")
        self.globals = kwa.get("globals", False) == True
        self.globals_prefix = kwa.get("globals_prefix", "") 
        print("Fold : loading from base %s setting globals %s globals_prefix %s " % (base, self.globals, self.globals_prefix)) 

        names = os.listdir(base)
        stems = []
        stamps = []

        for name in filter(lambda n:n.endswith(".npy") or n.endswith(".txt"),names):
            path = os.path.join(base, name)
            st = os.stat(path)
            stamp = datetime.datetime.fromtimestamp(st.st_ctime)

            stamps.append(stamp)

            is_npy = name.endswith(".npy")
            is_txt = name.endswith(".txt")
            stem = name[:-4]
            stems.append(stem)

            txt_dtype = "|S100" if stem.endswith("_meta") else np.object 
            a = np.load(path) if is_npy else np.loadtxt(path, dtype=txt_dtype, delimiter="\t") 

            # use non-present delim so lines with spaces do not cause errors
            #list(map(str.strip,open(path).readlines())) 
            setattr(self, stem, a ) 
            ashape = str(a.shape) if is_npy else len(a)    
            if self.globals:
                gstem = self.globals_prefix + stem
                globals()[gstem] = a 
                print(" %20s : %15s : %20s : %20s : %s " % (stem, gstem, ashape, str(stamp), name ))  
            else:
                print(" %20s : %15s : %20s :  %s " % (stem, ashape, str(stamp), name ))  
            pass
        pass

        min_stamp = min(stamps)
        max_stamp = max(stamps)
        dif_stamp = max_stamp - min_stamp 

        print(" min_stamp : %s " % str(min_stamp))
        print(" max_stamp : %s " % str(max_stamp))
        print(" dif_stamp : %s " % str(dif_stamp))

        assert dif_stamp.microseconds < 1e6, "stamp divergence detected microseconds %d : so are seeing mixed up results from multiple runs " % dif_stamp.microseconds 
        


        self.stems = stems
        self.stamps = stamps

    def desc(self, stem):
        a = getattr(self, stem)
        ext = ".txt" if a.dtype == 'O' else ".npy"
        path = os.path.join(self.base, "%s%s" % (stem,ext))
        return " %15s : %15s : %s " % ( stem, str(a.shape), path )

    def __repr__(self):
        return "\n".join(self.desc(stem) for stem in self.stems)    
  

if __name__  == '__main__':
    pass


