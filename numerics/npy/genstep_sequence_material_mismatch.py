#!/usr/bin/env python
"""

"""

import sys
import numpy as np
from env.numerics.npy.types import phc_, iflags_, imat_, iabmat_, ihex_, seqhis_, seqmat_, stc_, c2g_


def sequence(h,a,b, gm, qm):
    #np.set_printoptions(formatter={'int':lambda x:hex(int(x))})
   
    im = imat_()

    for i,p in enumerate(h[a:b]):
        ix = a+i
        his,mat = p[0],p[1]
        print " %7d : %16s %16s : %30s %30s : %10s %10s " % ( ix,ihex_(his),ihex_(mat), 
                   seqhis_(his), seqmat_(mat),
                   im[gm[ix]], im[qm[ix]] 
                  )    


if __name__ == '__main__':

    tag = 1 

    gs = stc_(tag)                                       # gensteps 
    seq = phc_(tag).reshape(-1,2)                        # flag,material sequence
    c2g = c2g_()                                      # chroma index to ggeo custom int
    im = iabmat_()

    gsnph = gs.view(np.int32)[:,0,3]                     # genstep: num photons 
    gspdg = gs.view(np.int32)[:,3,0]

    gsmat_c = gs.view(np.int32)[:,0,2]                     # genstep: material code  : in chroma material map lingo
    gsmat   = np.array( map(lambda _:c2g.get(_,-1), gsmat_c ), dtype=np.int32)   # translate chroma indices into ggeo custom ones


    p_gsmat   = np.repeat(gsmat, gsnph)                    # photon: genstep material code  
    p_seqmat = seq[:,1] & 0xF                             # first material in seqmat 
    off = np.arange(len(p_gsmat))[ p_gsmat != p_seqmat ]

    p_gsidx = np.repeat(np.arange(len(gsnph)), gsnph )    # photon: genstep index


    #n = len(sys.argv)
    #a = int(sys.argv[1]) if n > 1 else 0
    #b = int(sys.argv[2]) if n > 2 else a + 40
    #
    #print n,a,b
    #sequence(seq,a,b, p_gsmat, p_seqmat)

    for i in off:
        his,mat = seq[i,0],seq[i,1]
        seqhis = seqhis_(his)
        if 'MI' in seqhis:continue
        print " %7d : %16s %16s : %30s %30s : gs:%2s sq:%2s " % ( i,ihex_(his),ihex_(mat), 
                   seqhis, seqmat_(mat),
                   im[p_gsmat[i]], im[p_seqmat[i]] 
                  )    




   




