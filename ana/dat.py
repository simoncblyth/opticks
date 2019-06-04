#!/usr/bin/env python
"""
dat.py
========




"""

import os, logging, numpy as np
log = logging.getLogger(__name__)


class Dat(object):
    """
    For interactive exploration of dimension > 3  arrays
    """
    def __init__(self, a, ina=None, jna=None, kna=None ):
        self.a = a

        if ina is None:
            ina = np.arange(0, a.shape[0] ).astype("|S16")
        if jna is None:
            jna = np.arange(0, a.shape[1] ).astype("|S16")
        if kna is None:
            kna = np.arange(0, a.shape[2] ).astype("|S16")

        self.ina = ina
        self.kna = kna
        self.jna = jna

        self.ijk = 0,0,0 
        self.sli = slice(0,None,1)

    def __getitem__(self, sli):
        self.sli = sli
        return self

    def _get_d(self):
        return self.a[self.i,self.j,self.k][self.sli]
    d = property(_get_d)

    def __repr__(self):
        return "\n".join(map(repr, [self.ijk, self.name, self.d]))

    def _get_name(self):
        return ",".join([self.ina[self.i], self.jna[self.j], self.kna[self.k]])
    name = property(_get_name)

    def _set_i(self, _i):
        assert _i < self.a.shape[0]
        self._i = _i
    def _get_i(self): 
        return self._i
    i = property(_get_i, _set_i)

    def _set_j(self, _j):
        assert _j < self.a.shape[1]
        self._j = _j
    def _get_j(self): 
        return self._j
    j = property(_get_j, _set_j)

    def _set_k(self, _k):
        assert _k < self.a.shape[2]
        self._k = _k
    def _get_k(self): 
        return self._k
    k = property(_get_k, _set_k)

    def _set_ijk(self, *ijk):
        assert len(ijk) == 1 and len(ijk[0]) == 3
        self.i = ijk[0][0]
        self.j = ijk[0][1]
        self.k = ijk[0][2]
    def _get_ijk(self):
        return (self.i, self.j, self.k)
    ijk = property(_get_ijk, _set_ijk)






if __name__ == '__main__':
    from opticks.ana.main import opticks_main 
    ok = opticks_main()

    a = np.load(os.path.expandvars("$IDPATH/GBndLib/GBndLib.npy"))
    d = Dat(a, None, "omat osur isur imat".split(), "g0 g1".split())

    print(d)


