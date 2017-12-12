#!/usr/bin/env python

"""

Select the OptiX CUDA logging text in debugger, then

::

    simon:~ blyth$ pbpaste > /tmp/u.txt

    simon:cfg4 blyth$ ucf.py 1230
                                         boundary_burn : 0.00111702492 : 0.00111702491995  
                                            scattering : 0.5026473403 : 0.502647340298  
                                            absorption : 0.6015041471 : 0.601504147053  
                                               reflect :    0.93871 : 0.938713490963  
                                         boundary_burn : 0.753801465 : 0.753801465034  
                                            scattering : 0.9998467565 : 0.999846756458  
                                            absorption : 0.4380195737 : 0.438019573689  
                                                  rsa0 :   0.714032 : 0.71403157711  
                                                  rsa1 :   0.330404 : 0.330403953791  
                                                  rsa2 :   0.570742 : 0.570741653442  
                                                  rsa3 :   0.375909 : 0.37590867281  
                                                  rsa4 :   0.784978 : 0.784978330135  
                                         boundary_burn : 0.8926543593 : 0.892654359341  
                                            scattering : 0.4410631955 : 0.441063195467  
                                            absorption : 0.7737424374 : 0.773742437363  
                                               reflect :    0.55684 : 0.556839108467  
                                         boundary_burn : 0.775349319 : 0.775349318981  
                                            scattering : 0.7521412373 : 0.752141237259  
                                            absorption : 0.4120023847 : 0.412002384663  
                                               reflect :    0.28246 : 0.282463103533  
                                         boundary_burn : 0.4324976802 : 0.432497680187  
                                            scattering : 0.9078488946 : 0.907848894596  
                                            absorption : 0.9121392369 : 0.912139236927  
                                               reflect :    0.20181 : 0.201808556914  
                                         boundary_burn : 0.7953493595 : 0.795349359512  
                                            scattering : 0.4842039943 : 0.484203994274  
                                            absorption : 0.09354860336 : 0.0935486033559  
                                               surface :     0.7505 : 0.750533461571  
                                          surface_burn :     0.9462 : 0.946246266365  





"""
import os, logging, sys, re, numpy as np
log = logging.getLogger(__name__)


class UCF(object):
    MKR = "u_"
    PTN = re.compile("u_(\S*):\s*(\S*)\s*")

    def __init__(self, path, xrng):
        self.ulab = []
        self.uval = []
        self.xrng = xrng
        self.parse(path) 
        
    def parse(self, path):
        for line in filter(lambda line:line.find(self.MKR) > -1, file(path).readlines()):
            m = self.PTN.search(line)
            assert m, " fail for line:%s " % line 
            self.ulab.append(m.group(1))
            self.uval.append(m.group(2))
        pass

    def __repr__(self):
        assert len(self.ulab) == len(self.uval)
        assert len(self.ulab) < len(self.xrng)

        return "\n".join(map(lambda _:"%50s : %10s : %10s  " % (self.ulab[_], self.uval[_], self.xrng[_]), range(len(self.ulab))))


if __name__ == '__main__':

    

    path = "/tmp/u.txt"
    prng = "$TMP/TRngBufTest.npy"

    assert len(sys.argv) > 1
    idx = int(sys.argv[1])


    log.info( "idx : %d " % idx ) 

    rng = np.load(os.path.expandvars(prng))
    xrng = rng[idx].ravel()

    ucf = UCF(path, xrng)

    print ucf


   
