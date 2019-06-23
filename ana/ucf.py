#!/usr/bin/env python
"""
ucf.py : Comparing Opticks and Geant4 random consumption
=============================================================

Compares the randoms consumed by Geant4 and Opticks and the
positions where the consumption happens.  Relies on mask 
running of single photons 

::

    ucf.py 1230
    UCF_PINDEX_DEV=-1 ucf.py 1230
    UCF_PINDEX_DEV=1230 ucf.py 1230


Parse the kernel print log in a u_rng centric fashion::

     tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog -DD   

          ## write kernel pindexlog for photon 1230

    ucf.py 1230

          ## parse the log and compare with expected rng sequence


NB this is invoked from CRandomEngine::preTrack when using --mask option, 
as printindexlog stdout redirection borks stdout for subprocesses 
this script must not write to stdout. 

cfg4/CRandomEngine.cc::

    367         const char* cmd = BStr::concat<unsigned>("ucf.py ", mask_index, NULL );
    368         LOG(info) << "CRandomEngine::preTrack : START cmd \"" << cmd << "\"";
    369         int rc = SSys::run(cmd) ;  // NB must not write to stdout, stderr is ok though 
    370         assert( rc == 0 );
    371         LOG(info) << "CRandomEngine::preTrack : DONE cmd \"" << cmd << "\"";


"""
from __future__ import print_function
import os, sys, re


def print_(s):
    stream = sys.stderr
    print(s, file=stream)


class U(object):

    XRNG = []   # set by UCF.__init__ 

    @classmethod
    def find(cls, u, tolerance=1e-6):
        """
        :return: index of u within XRNG sequence, or None if not found
        """ 
        idxf = None
        for i in range(len(cls.XRNG)):
            if abs(u-cls.XRNG[i]) < tolerance:
                idxf = i
                break 
            pass
        pass
        return idxf

    def __init__(self, idx, lab, val, lines):
        """
        :param idx: index into XRNG sequence
        :param lab: 
        :param val:
        :param lines:  
        """
        cls = self.__class__ 
        self.idx = idx
        self.lab = lab
        self.val = val
        self.fval = float(val)

        assert idx < len(cls.XRNG)
        xval = cls.XRNG[idx]

        idxf = cls.find(self.fval) 
        idxd = idxf - idx 

        self.idxf = idxf
        self.idxd = idxd

        self.xval = xval 
        self.lines = lines[:]      ## must copy, not reference
        self.tail = []

    def _get_hdr(self):
        fval = "%.9f" % self.fval
        xval = "%.9f" % self.xval 
        mrk = "    " if fval == xval else "%+3d*" % self.idxd 
        return " [%3d|%3d] %50s : %2s : %s : %s : %d " % ( self.idx, self.idxf, self.lab, mrk, fval, xval , len(self.lines) ) 
    hdr = property(_get_hdr)

    def __str__(self):
        return "\n".join(self.lines + [self.hdr,""] + self.tail) 

    def __repr__(self):
        return self.hdr 


class UCF(list):
    @classmethod
    def rngpath(cls):
        return os.path.expandvars("$TMP/TRngBufTest.npy" )
    @classmethod
    def rngpathtxt(cls, pindex):
        return os.path.expandvars("$TMP/TRngBufTest_%s.txt" % pindex )
    @classmethod
    def printlogpath(cls, pindex):
        """
        :return: path to single photon log, obtained by redirection of OptiX output stream 
        """
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

    def __init__(self, pindex):
        """
        :param pindex: photon record index 

        1. collects random consumption reported in OptiX single photon log into this list
        2. loads the expected sequence of randoms for this photon  

        """
        list.__init__(self)

        evar = "UCF_PINDEX_DEV"
        upindex = int(os.environ.get(evar, pindex))
        if upindex != pindex:
            print_("WARNING evar active %s " % evar )
        pass
        path = self.printlogpath(upindex)
        xrng = self.loadrngtxt(pindex)

        print_("path %s " % path )


        U.XRNG = xrng 

        self.pindex = pindex
        self.path = path 
        self.parse(path) 


    PTN = re.compile("u_(\S*):\s*(\S*)\s*")

    def parse(self, path):
        """
        Parses the single photon log, collecting consumption lines 
        into this list as U instances. 
        """  
        self.lines = map(lambda line:line.rstrip(),file(path).readlines())
        curr = []
        for i, line in enumerate(self.lines):
            m = self.PTN.search(line)
            #print "%2d : %s" % ( i, line)
            curr.append(line)
            if m is None or line[0] == "#": continue

            sname = m.group(1)
            srng = m.group(2)

            idx = len(self)
            u = U(idx, sname, srng, curr )
            curr[:] = []  ## clear collected lines

            self.append(u)
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

    rng = np.load(UCF.rngpath())
    xrng = rng[pindex].ravel()
    #print_(str(xrng)) 

    ## workaround lack of numpy in system python used by lldb
    ## by writing rng for the pindex to txt file 
    trng = UCF.rngpathtxt(pindex)
    np.savetxt(trng, xrng, delimiter=",")   
    xrng2 = np.array(UCF.loadrngtxt(pindex), dtype=np.float64)
    #print_(xrng2)
    assert np.all(xrng2 == xrng)

    ucf = UCF( pindex )

    #print_(str(ucf))
    print_(repr(ucf))

   
