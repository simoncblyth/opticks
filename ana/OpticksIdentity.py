#!/usr/bin/env python
"""


In [80]: trpo[trpo[:,1] == 5]                                                                                                                                                                  
Out[80]: 
array([[83886080,        5,        0,        0],
       [83886081,        5,        0,        1],
       [83886082,        5,        0,        2],
       ...,
       [84057858,        5,      671,        2],
       [84057859,        5,      671,        3],
       [84057860,        5,      671,        4]], dtype=uint32)


"""
import numpy as np

class OpticksIdentity(object):
    """
    cf okc/OpticksIdentity.cc
    """
    @classmethod
    def Encode(cls, ridx, pidx, oidx):
        if ridx > 0:
            assert (ridx & 0xff) == ridx
            assert (pidx & 0xffff) == pidx
            assert (oidx & 0xff) == oidx
            return (ridx << 24) | (pidx << 8) | (oidx << 0)
        else:
            assert (ridx & 0xff) == ridx
            assert pidx == 0 
            assert (oidx & 0xffffff) == oidx 
            return (ridx << 24) | (oidx << 0)
        pass
    @classmethod
    def Decode(cls, tid):
        ridx = ( tid >> 24 ) & 0xff 
        pidx = np.where( ridx == 0,                       0, ( tid >>  8 ) & 0xffff )
        oidx = np.where( ridx == 0, ( tid >> 0 ) & 0xffffff, ( tid >> 0  ) & 0xff   )
        return ridx,pidx,oidx

    @classmethod
    def NRPO(cls, tid):
        """
        Decode the triplet identifier to show nidx/ridx/pidx/oidx (node/repeat/placement/offset-idx)
        of all volumes, see okc/OpticksIdentity::Decode::
            In [44]: nrpo[nrpo[:,1] == 5]                                                                                                                                                                    
            Out[44]: 
            array([[ 3199,     5,     0,     0],
                   [ 3200,     5,     0,     1],
                   [ 3201,     5,     0,     2],
                   ...,
                   [11410,     5,   671,     2],
                   [11411,     5,   671,     3],
                   [11412,     5,   671,     4]], dtype=uint32)

        """
        nidx = np.arange(len(tid), dtype=np.uint32)
        ridx,pidx,oidx = cls.Decode(tid)  
        nrpo = np.zeros( (len(tid),4), dtype=np.uint32 )
        nrpo[:,0] = nidx
        nrpo[:,1] = ridx
        nrpo[:,2] = pidx
        nrpo[:,3] = oidx
        return nrpo

     
if __name__ == '__main__':
    import os, numpy as np
    from opticks.ana.key import keydir
    avi = np.load(os.path.join(keydir(),"GNodeLib/all_volume_identity.npy"))
    tid = avi[:,1] 
    nrpo = OpticksIdentity.NRPO(tid)
    


