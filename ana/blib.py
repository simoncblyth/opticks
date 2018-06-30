#!/usr/bin/env python

import sys, os, numpy as np
tx_load = lambda _:map(str.strip, open(_).readlines())

class BLib(object):
    def path(self, rel):
        return os.path.join(self.idpath, rel)

    @classmethod
    def make(cls, path):
        return cls(cls.find_idpath(path))

    @classmethod
    def find_idpath(cls, path):
        """
        Convert any absolute path inside the idpath into the idpath 
        """
        elem = path.split("/")
        elen = map(len,elem)
        digp = elen.index(32)   # digest has length of 32 
        idpath = "/".join(elem[:digp+2])  # one past the digest 
        return idpath

    def __init__(self, idpath):
        self.idpath = idpath
        blib = np.load(self.path("GBndLib/GBndLibIndex.npy"))
        mlib = tx_load(self.path("GItemList/GMaterialLib.txt"))
        slib = tx_load(self.path("GItemList/GSurfaceLib.txt"))
        self.blib = blib
        self.mlib = mlib
        self.slib = slib
    def mname(self, idx):
        return self.mlib[idx] if idx < len(self.mlib) else ""
    def sname(self, idx):
        return self.slib[idx] if idx < len(self.slib) else ""
    def bname(self, idx):
        omat, osur, isur, imat = self.blib[idx] 
        return "/".join(
                    [ self.mname(omat), 
                      self.sname(osur), 
                      self.sname(isur), 
                      self.mname(imat) ] )

    def __repr__(self):
        return " nbnd %3d nmat %3d nsur %3d " % ( len(self.blib), len(self.mlib), len(self.slib))
    def __str__(self):
        return "\n".join([repr(self)] +  map(lambda _:"%3d : %s " % ( _, self.bname(_)) , range(len(self.blib))))


if __name__ == '__main__':

    bdir = sys.argv[1] if len(sys.argv) > 1 else os.environ["IDPATH"]
    blib = BLib.make(bdir)
    print blib

     
