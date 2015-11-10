#!/usr/bin/env python
import os
import numpy as np

idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )

M_REFRACTIVE_INDEX = 0
M_ABSORPTION_LENGTH = 1
M_SCATTERING_LENGTH = 2
M_REEMISSION_PROB = 3

S_DETECT = 0 
S_ABSORB = 1 
S_REFLECT_SPECULAR = 2 
S_REFLECT_DIFFUSE = 3 

B_OMAT = 0
B_OSUR = 1
B_ISUR = 2
B_IMAT = 3


class PropLib(object):
    def __init__(self, kls="GMaterialLib"):
        fpath=idp_("%(kls)s/%(kls)s.npy" % locals())
        npath=idp_("GItemList/%(kls)s.txt" % locals())
        data = np.load(fpath)
        names = map(lambda _:_[:-1],file(npath).readlines())
        assert len(names) == data.shape[0]
        pass
        self.data = data
        self.names = names
        self.domain = np.linspace(60.,810., 39)

    def index(self, name):
        return self.names.index(name)

    def interp(self, name, wavelengths, k=0):
        idx = self.names.index(name)
        return np.interp( wavelengths, self.domain, self.data[idx][:,k] ) 
 
    def __call__(self, name):
        idx = self.names.index(name)
        return self.data[idx]
   

if __name__ == '__main__':
    mlib = PropLib("GMaterialLib") 
    slib = PropLib("GSurfaceLib") 
    blib = PropLib("GBndLib") 

    idx = mlib.index("Water")

    m1 = "Water"
    wavelength = 442  
    ri = mlib.interp(m1, wavelength, M_REFRACTIVE_INDEX) 

    print "m1 %s wl %s ri %s " % (m1, wavelength, ri)




