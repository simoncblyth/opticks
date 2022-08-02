#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

class fold(object):
    pass

class sfreq(object):
    @classmethod
    def CreateFromArray(cls, a):
        usub, nsub = np.unique(a, return_counts=True )
        f = fold()
        f.key = usub
        f.val = nsub 
        return cls(f, sort=True) 

    @classmethod
    def ArrayFromFile(cls, base=None, name="subs.txt"):
        if base is None:
            base = os.environ.get("FOLD", None)
        pass
        path = os.path.join(base, name)
        a = np.loadtxt(path, dtype="|S32" )  
        return a 

    @classmethod
    def CreateFromFile(cls, base=None, name="subs.txt"):
        a = cls.ArrayFromFile(base=base, name=name)
        return cls.CreateFromArray(a)

    def __init__(self, f, sort=True):
        order = np.argsort(f.val)[::-1] if sort else slice(None)
        okey = f.key[order]  
        oval = f.val[order]
        subs = list(map(lambda _:_.decode("utf-8"), okey.view("|S32").ravel() ))  
        vals = list(map(int, oval)) 

        self.f = f   # without the sort 
        self.okey = okey 
        self.oval = oval 
        self.subs = subs
        self.vals = vals
        self.order = order


    def find_index(self, key):
        key = key.encode() if type(key) is str else key
        assert type(key) is bytes
        ii = np.where( self.okey.view("|S32") == key )[0]     ## hmm must use okey to feel the sort
        assert len(ii) == 1
        return int(ii[0])

    def desc_key(self, key):
        idx = self.find_index(key)
        return self.desc_idx(idx)

    def desc_idx(self, idx):
        return "sf %3d : %7d : %s." % (idx, self.vals[idx], self.subs[idx]) 

    def __repr__(self):
        return "\n".join(self.desc_idx(idx) for idx in range(len(self.subs))) 

    def __str__(self):
        return str(self.f)


if __name__ == '__main__':
    sf = sfreq.CreateFromFile()
    print(repr(sf))





