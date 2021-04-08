#!/usr/bin/env python
""" 
Using approx hc eV to nm conversion of 1240 as it seems that was done upstream, 
hence using this approx value will actually be better as it should 
give the measurement nm from LS group.

Rather than using the more precise 1239.8418754199977 which 
will give nm slightly off 
"""
#hc_eVnm=1239.8418754199977  # h_Planck*c_light*1e12    
hc_eVnm=1240.0

import sys, os, numpy as np, logging, argparse, inspect
log = logging.getLogger(__name__)
tx_load = lambda _:list(map(str.strip, open(_).readlines())) # py3 needs the list, otherwise stays as map 

from opticks.ana.key import keydir
KEYDIR = keydir()

class GMaterialLib(object):

    RINDEX = (0,0)
    ABSLEN = (0,1)
    RAYLEIGH = (0,2)
    REEMPROB = (0,3)
        
    GROUPVEL = (1,0)
    SPARE1 = (1,1) 
    SPARE2 = (1,2) 
    SPARE3 = (1,3) 

    @classmethod 
    def Qwn(cls):
        qwn = {}
        for k, v in inspect.getmembers(cls): 
            if not type(v) is tuple: continue
            qwn[k] = v 
        pass
        return qwn 

    def __init__(self):
        self.qwn = self.Qwn()
        names = tx_load(os.path.join(KEYDIR, "GItemList/GMaterialLib.txt"))
        props = np.load(os.path.join(KEYDIR, "GMaterialLib/GMaterialLib.npy"))
        assert len(names) == len(props) 
        self.names = names
        self.props = props

        # TODO:domain should be persisted in geocache, not assumed fixed
        nm = np.linspace(60.,820.,39)   # standard wavelength domain with increasing wavelenth (nm)
        ev = hc_eVnm/nm 

        self.nm = nm
        self.ev = ev
         

    def material_idx(self, mat_name):
        return self.names.index(mat_name)

    def __call__(self, spec):
        mat_name, qty_name = spec.split(".")
        return self.prop(mat_name, qty_name)

    def prop(self, mat_name, qty_name):
        """
        :param mat_name: material name eg LS 
        :param qty_name: quantity name eg RINDEX, ABSLEN, RAYLEIGH, REEMPROB, GROUPVEL 
        :return a: np.ndarray property as function of wavelength eg of shape (39,) 
        """
        assert mat_name in self.names
        midx = self.material_idx(mat_name)

        assert qty_name in self.qwn
        qgrp,qidx =  self.qwn.get(qty_name)       
        a = self.props[midx,qgrp,:,qidx] 
        return a

    def __repr__(self):
        return "GMaterialLib %d " % len(self.names) 
    def __str__(self):
        return "\n".join([repr(self)] + self.names)


if __name__ == '__main__':
    mlib = GMaterialLib()
    #print(mlib)

    rils = mlib.prop("LS",      "RINDEX")
    riac = mlib.prop("Acrylic", "RINDEX")
    riai = mlib.prop("Air",     "RINDEX")
    ripy = mlib.prop("Pyrex",   "RINDEX")
    riwt = mlib.prop("Water",   "RINDEX")

    assert np.all( riai == riai[0] )   

    for mat_name in mlib.names:
        ri = mlib.prop(mat_name, "RINDEX")
        if np.all(ri == 1.): continue 
        print(mat_name)
        print(ri)
    pass


