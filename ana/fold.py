#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

class Fold(object):
    @classmethod
    def Load(cls, *args, **kwa):
        base = os.path.join(*args)
        base = os.path.expandvars(base) 
        return cls(base, **kwa) if os.path.isdir(base) else None

    def __init__(self, base, **kwa):
        self.base = base
        self.kwa = kwa 
        self.globals = kwa.get("globals", False) == True
        self.globals_prefix = kwa.get("globals_prefix", "") 
        print("Fold : loading from base %s setting globals %s globals_prefix %s " % (base, self.globals, self.globals_prefix)) 

        names = os.listdir(base)
        for name in filter(lambda n:n.endswith(".npy") or n.endswith(".txt"),names):
            path = os.path.join(base, name)
            is_npy = name.endswith(".npy")
            is_txt = name.endswith(".txt")
            stem = name[:-4]
            a = np.load(path) if is_npy else list(map(str.strip,open(path).readlines())) 
            setattr(self, stem, a ) 
            ashape = str(a.shape) if is_npy else len(a)    

            if self.globals:
                gstem = self.globals_prefix + stem
                globals()[gstem] = a 
                print(" %10s : %15s : %15s : %s " % (stem, gstem, ashape, path ))  
            else:
                print(" %10s : %15s : %s " % (stem, ashape, path ))  
            pass
        pass

if __name__  == '__main__':
    pass


