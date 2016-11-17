#!/usr/bin/env python
"""
PropLib : Geocache Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*PropLib* provides access to optical properties of materials and surfaces 
as a function of wavelength.

Material properties::

    REFRACTIVE_INDEX 
    ABSORPTION_LENGTH 
    SCATTERING_LENGTH 
    REEMISSION_PROB 

Surface properties::

    DETECT 
    ABSORB 
    REFLECT_SPECULAR 
    REFLECT_DIFFUSE 


Example data shapes::

    In [5]: mlib.data.shape
    Out[5]: (38, 2, 39, 4)

    In [6]: slib.data.shape
    Out[6]: (48, 2, 39, 4)

    In [7]: blib.data.shape
    Out[7]: (123, 4, 2, 39, 4)


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_environment, stamp_
from opticks.ana.dat import Dat

idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )




class Bnd(object):
    def __init__(self, names):
        nam = np.zeros( (len(names),4 ), dtype="|S64")
        for i,name in enumerate(names):
            nam[i] = name.split("/")
        pass
        self.nam = nam

    def oms(self, mat):
        """bnd indices of omat"""
        return np.where(self.nam[:,0] == mat)[0]   

    def ims(self, mat):
        """bnd indices of imat"""
        return np.where(self.nam[:,3] == mat)[0]   



class PropLib(object):

    # 1st set of 4 [0] 
    M_REFRACTIVE_INDEX = 0
    M_ABSORPTION_LENGTH = 1
    M_SCATTERING_LENGTH = 2
    M_REEMISSION_PROB = 3

    L_GROUP_VELOCITY = 0

    # 2nd set of 4 [0] startswith GROUPVEL, currently not used

    S_DETECT = 0 
    S_ABSORB = 1 
    S_REFLECT_SPECULAR = 2 
    S_REFLECT_DIFFUSE = 3 

    B_OMAT = 0
    B_OSUR = 1
    B_ISUR = 2
    B_IMAT = 3

    def __init__(self, kls="GMaterialLib", dpath=None):

        if dpath is None:
            dpath=idp_("%(kls)s/%(kls)s.npy" % locals())
        else:
            log.warning("override dpath to %s " % dpath)
            dpath = os.path.expandvars(dpath)
        pass

        npath=idp_("GItemList/%(kls)s.txt" % locals())
        names = map(lambda _:_[:-1],file(npath).readlines())

        self.kls = kls
        paths = []
        paths.append(dpath)
        paths.append(npath)

        data_ = np.load(dpath)

        if len(names) != data_.shape[0]:
            data = data_.reshape(-1,4,2,data_.shape[0],4)
            log.warning("reshaped %s from  %r -> %r  " % (dpath, data_.shape, data.shape) )
        else:
            data = data_
        pass 


        assert len(names) == data.shape[0]
        pass
        self.data = data
        self.names = names
        #self.domain = np.linspace(60.,810., 39)
        self.domain = np.linspace(60.,820., 39)

        if kls == "GBndLib":
            opticalpath = idp_("%(kls)s/%(kls)sOptical.npy" % locals())
            self.optical = np.load(opticalpath)
            paths.append(opticalpath)
            self.dat = Dat(data, names, "omat osur isur imat".split(), "g0 g1".split() )
            self.bnd = Bnd(names) 
        pass
        self.paths = paths

    brief = property(lambda self:"%s" % ( self.kls ))

    def __repr__(self):
        return "\n".join([self.brief] + self.paths + map(stamp_, self.paths)) 

    def index(self, name):
        return self.names.index(name)

    def interp(self, name, wavelengths, k=0):
        idx = self.names.index(name)
        return np.interp( wavelengths, self.domain, self.data[idx][0][:,k] ) 
 
    def __call__(self, name):
        idx = self.names.index(name)
        return self.data[idx]
   


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    mlib = PropLib("GMaterialLib") 
    slib = PropLib("GSurfaceLib") 
    blib = PropLib("GBndLib") 

    idx = mlib.index("Water")

    m1 = "Water"
    wavelength = 442  
    ri = mlib.interp(m1, wavelength, mlib.M_REFRACTIVE_INDEX) 

    print "m1 %s wl %s ri %s " % (m1, wavelength, ri)


    op = blib.optical


