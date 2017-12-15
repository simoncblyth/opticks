#!/usr/bin/env python
"""
ucf.py
=======

Parse the kernel print log in a u_rng centric fashion::

     tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog -DD   

          ## write kernel pindexlog for photon 1230

    ucf.py 1230

          ## parse the log and compare with expected rng sequence



"""
from __future__ import print_function
import os, sys, re



class U(object):
    def __init__(self, idx, lab, val, xval, lines):
        self.idx = idx
        self.lab = lab
        self.val = val
        self.xval  = xval 
        self.lines = lines[:]   ## must copy 
        self.tail = []

    def _get_fval(self):
        return float(self.val)
    fval = property(_get_fval)

    def _get_hdr(self):
        fval = "%.9f" % self.fval
        xval = "%.9f" % self.xval 
        mrk = "  " if fval == xval else "**"
        return " [%3d] %50s : %15s : %2s : %s : %s : %d " % ( self.idx, self.lab, self.val, mrk, fval, xval , len(self.lines) ) 
    hdr = property(_get_hdr)

    def __str__(self):
        return "\n".join(self.lines + [self.hdr,""] + self.tail) 

    def __repr__(self):
        return self.hdr 


class UCF(list):
    MKR = "u_"
    PTN = re.compile("u_(\S*):\s*(\S*)\s*")


    @classmethod
    def rngpath(cls):
        return os.path.expandvars("$TMP/TRngBufTest.npy" )
    @classmethod
    def rngpathtxt(cls, pindex):
        return os.path.expandvars("$TMP/TRngBufTest_%s.txt" % pindex )
    @classmethod
    def printlogpath(cls, pindex):
        return os.path.expandvars("$TMP/ox_%s.log" % pindex )

    @classmethod
    def loadrngtxt(cls, pindex):
        """
        workaround lldb python failing to import numpy 
        """
        trng_ = cls.rngpathtxt(pindex) 
        trng = os.path.expandvars(trng_)
        assert os.path.exists(trng), (trng, trng_, "non-existing-trng") 
        return map(float, file(trng).readlines())


    def __init__(self, pindex ):
        list.__init__(self)

        path = self.printlogpath(pindex)
        xrng = self.loadrngtxt(pindex)

        self.pindex = pindex
        self.path = path 
        self.xrng = xrng
        self.parse(path) 
        
    def parse(self, path):

        self.lines = map(lambda line:line.rstrip(),file(path).readlines())

        curr = []
        for i, line in enumerate(self.lines):
            m = self.PTN.search(line)

            #print "%2d : %s" % ( i, line)

            curr.append(line)

            if m is None: continue

            idx = len(self)
            assert idx < len(self.xrng)
            xval = self.xrng[idx]

            u = U(idx, m.group(1), m.group(2), xval, curr )
            self.append(u)

            curr[:] = []
        pass
        self[-1].tail = curr[:]


    def _get_hdr(self):
        return " %7d : %s  " % ( self.pindex, self.path )
    hdr = property(_get_hdr)   

    def __str__(self):
        return "\n".join([self.hdr]+map(str, self))

    def __repr__(self):
        return "\n".join([self.hdr]+map(repr, self))


if __name__ == '__main__':


    pindex = int(sys.argv[1]) if len(sys.argv) > 1 else 1230

    import numpy as np


    stream = sys.stderr

    rng = np.load(UCF.rngpath())
    xrng = rng[pindex].ravel()
    print(str(xrng), file=stream) 

    trng = UCF.rngpathtxt(pindex)
    np.savetxt(trng, xrng, delimiter=",")

    xrng2 = UCF.loadrngtxt(pindex)
    print(str(xrng2), file=stream)


    ucf = UCF( pindex )

    print(str(ucf), file=stream)
    print(repr(ucf), file=stream)

   
