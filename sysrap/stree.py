#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

class sfreq(object):
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
        return " %3d : %7d : %s " % (idx, self.vals[idx], self.subs[idx]) 

    def __repr__(self):
        return "\n".join(self.desc_idx(idx) for idx in range(len(self.subs))) 

    def __str__(self):
        return str(self.f)


class snode(object):
    DTYPE = [('index', '<i4'), 
             ('depth', '<i4'), 
             ('sibdex', '<i4'), 
             ('parent', '<i4'), 
             ('num_child', '<i4'), 
             ('first_child', '<i4'), 
             ('next_sibling', '<i4'), 
             ('lvid', '<i4'), 
             ('copyno', '<i4')] 

    @classmethod
    def RecordsFromArrays(cls, a):
        """
        :param a: nodes ndarray
        :return nds: np.recarray
        """
        nds = np.core.records.fromarrays(a.T, dtype=cls.DTYPE )
        return nds 

    @classmethod
    def Desc(cls, rec):
        """
        :param rec:  single rec eg obtained by nds[0] 
        :return str: formatted description
        """
        return "snode ix:%7d dh:%2d sx:%5d pt:%7d nc:%5d fc:%7d ns:%7d lv:%3d cp:%7d " % tuple(rec)
                   

class stree(object):
    def __init__(self, f):
        sff = Fold.Load(f.base,"subs_freq",  symbol="sf") 
        sf = sfreq(sff)

        self.sf = sf
        self.f = f 

        self.nds = snode.RecordsFromArrays(f.nds)

    def get_children(self, nidx):
        """
        This is a direct translation of the C++ stree::get_children 
        which will inevitably be slow in python. 
        HMM: how to do this in a more numpy way ?
        """
        children = []
        nd = self.nds[nidx]
        assert nd.index == nidx
        ch = nd.first_child
        while ch > -1:
           child = self.nds[ch] 
           assert child.parent == nd.index
           children.append(child.index)
           ch = child.next_sibling
        pass
        return children

    def get_progeny_r(self, nidx, progeny):
        """
        :param nidx: node index
        :param progeny: in/out list of node indices

        Recursively collects node indices of progeny beneath nidx, 
        not including nidx itself.  
        """
        children = self.get_children(nidx)
        progeny.extend(children)
        for nix in children:
            self.get_progeny_r(nix, progeny)
        pass

    def get_progeny(self, nidx):
        progeny = []
        self.get_progeny_r(nidx, progeny)
        return progeny  

    def get_soname(self, nidx):
        lvid = self.get_lvid(nidx)
        return self.f.soname[lvid] if lvid > -1 and lvid < len(self.f.soname) else None ; 
 
    def get_sub(self, nidx):
        return self.f.subs[nidx] if nidx > -1 else None ; 
 
    def get_depth(self, nidx):
        return self.nds[nidx].depth if nidx > -1 else  -1 ; 
   
    def get_parent(self, nidx): 
        return self.nds[nidx].parent if nidx > -1 else  -1 ; 

    def get_lvid(self, nidx): 
        return self.nds[nidx].lvid if nidx > -1 else  -1 ; 

    def get_transform(self, nidx):
        return self.f.trs[nidx] if nidx > -1 and nidx < len(self.f.trs) else None

    def get_ancestors(self, nidx):
        ancestors = [] 
        parent = self.get_parent(nidx)
        while parent > -1:
            ancestors.append(parent) 
            parent = self.get_parent(parent)
        pass
        ancestors.reverse()
        return ancestors 

    def __repr__(self):
        return repr(self.f)
    def __str__(self):
        return str(self.f)


if __name__ == '__main__':
    f = Fold.Load(symbol="f")

    st = stree(f)
    print(repr(st))

    nidx = os.environ.get("NIDX", 13)

    progeny = st.get_progeny(nidx)
    print(progeny) 
  
    ancestors = st.get_ancestors(nidx)
    print(ancestors)

    for nix in ancestors:
        print(snode.Desc(st.nds[nix]))
    pass








