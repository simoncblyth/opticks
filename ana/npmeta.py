#!/usr/bin/env python
"""
NPMeta.py
===========

Parsing metadata lines from NP.hh 

TODO: introspective listing of keys, rather than current needing to know whats there 

"""

import os, logging
from collections import OrderedDict as odict 
import numpy as np
log = logging.getLogger(__name__)


class NPMetaCompare(object):
    def __init__(self, am, bm):

        ak = am.d.keys() 
        bk = bm.d.keys() 
        kk = ak if ak == bk else list(set(list(ak)+list(bk))) 
        skk = np.array( list(map(lambda _:"%30s"%_, kk )), dtype="|S30" )
        tab = np.zeros( [len(kk),2], dtype=np.uint64 ) 
        lines = [] 

        hfmt = "%-30s : %7s : %7s : %7s : %7s : %s "
        vfmt = "%-30s : %7d : %7d : %7.3f : %7.3f : %7.3f"
        lines.append(hfmt % ("key", "a", "b", "a/b", "b/a", "(a-b)^2/(a+b)" ))  

        for i, k in enumerate(kk):

            if k == "": continue
            if k == "SEvt__TimerDone": continue
            al = am.d.get(k,[])
            bl = bm.d.get(k,[])

            av = al[0] if len(al) == 1 else 0
            bv = bl[0] if len(bl) == 1 else 0

            #av_bv = 0 if bv == 0. else av/bv
            #bv_av = 0 if av == 0. else bv/av
            #c2    = 0 if av+bv == 0 else (av-bv)*(av-bv)/(av+bv)
            #lines.append(vfmt % ( k, av, bv, av_bv, bv_av, c2 ))

            tab[i,0] = av
            tab[i,1] = bv
        pass

        stab = np.c_[skk, tab]


        self.am = am
        self.bm = bm
        self.kk = kk
        self.skk = skk
        self.tab = tab
        self.lines = lines
        self.stab = stab

    def __str__(self):
        return "\n".join(self.lines)

    def __repr__(self):
        lines = []
        lines.append("skk")
        return "\n".join(lines)
  

class NPMeta(object):
    ENCODING = "utf-8"
    @classmethod
    def Compare(cls, am, bm):
        cfm = NPMetaCompare(am,bm)
        return cfm  

    @classmethod
    def AsDict(cls, lines):
        d = odict()
        key = "" 
        d[key] = []
        for line in lines:
            dpos = line.find(":") 
            if dpos > -1:
                key = line[:dpos] 
                d[key] = []
                val = line[dpos+1:] 
            else:
                val = line 
            pass
            d[key].append(val)
        pass    
        return d 

 
    @classmethod
    def Load(cls, path):
        name = os.path.basename(path)
        lines = open(path, "r").read().splitlines()
        return cls(lines) 

        #txt_dtype = "|S100" if stem.endswith("_meta") else np.object 
        #t = np.loadtxt(path, dtype=txt_dtype, delimiter="\t") 
        #if t.shape == (): ## prevent one line file behaving different from multiline 
        #    a = np.zeros(1, dtype=txt_dtype)
        #    a[0] = cls(str(t))   
        #else:
        #    a = cls(t)     
        #pass

    def __init__(self, lines):
        self.lines = lines  
        self.d = self.AsDict(lines)
           
    def __len__(self):
        return len(self.lines)  

    def find(self, k, fallback=None):
        return self.d.get(k, fallback)

    def __getattr__(self, k):
        if not k in self.d:
             raise AttributeError("No attribute %s " % k) 
        return self.find(k)

    def __getitem__(self, idx):
        """
        item access useful for simple lists of names, not metadata dicts 
        """
        return self.lines[idx] 

    def oldfind(self, k_start, fallback=None, encoding=ENCODING):
        meta = self.meta
        ii = np.flatnonzero(np.char.startswith(meta, k_start.encode(encoding)))  
        log.debug( " ii %s len(ii) %d  " % (str(ii), len(ii)) )
        ret = fallback 
        if len(ii) == 1:
            i = ii[0]
            line = meta[i].decode(encoding)
            ret = line[len(k_start):]
            log.debug(" line [%s] ret [%s] " % (line,ret) )
        else:
            log.debug("did not find line starting with %s or found more than 1" % k_start) 
        pass
        return ret 

    def __repr__(self):
        return "\n".join(self.lines)
    def __str__(self):
        return repr(self.d)



def test_load():

    path = "/tmp/t_meta.txt"
    multiline = "hello:world\nmoi:red\nmidx:green\nmord:blue\niidx:grey\nTOPLINE:yellow\nBOTLINE:red\n"
    oneline = "hello:world\n"
    test = oneline
    open(path, "w").write(test)


    pm = NPMeta.Load(path)

    moi = pm.find("moi:")
    midx = pm.find("midx:")
    mord = pm.find("mord:")
    iidx = pm.find("iidx:")
    print(" moi:[%s] midx:[%s] mord:[%s] iidx:[%s] " % (moi, midx, mord, iidx) )

    TOPLINE = pm.find("TOPLINE:")
    BOTLINE = pm.find("BOTLINE:")

    print(" TOPLINE:[%s] " % TOPLINE )
    print(" BOTLINE:[%s] " % BOTLINE )



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    lines = ['PV:nnvt_body_phys',
             'nnvt_inner1_phys',
             'nnvt_inner2_phys',
             'nnvt_tube_phy',
             'nnvt_edge_phy',
             'hama_body_phys',
             'nnvt_plate_phy',
             'hama_inner1_phys',
             'hama_inner2_phys',
             'hama_outer_edge_phy',
             'hama_plate_phy',
             'hama_dynode_tube_phy',
             'hama_inner_ring_phy',
             'MLV:nnvt_log',
             'nnvt_body_log',
             'nnvt_inner2_log',
             'hama_log',
             'hama_body_log',
             'hama_inner2_log']


    m = NPMeta(lines)
    print(m.d)
     



    


