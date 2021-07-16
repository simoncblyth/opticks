#/usr/bin/env python
"""
ckcf.py 
=======

::

    ipython -i ckcf.py 


Compare 3 ways to generate Cerenkov wavelengths:

A:QCtxTest 
    CUDA impl using wavelength sampling reciprocalized to mimic energy sampling  
    qu ; om ; QCtxTest 
    randoms are direct from curand, which should match the TRngTest precooked ones 
    use by the other two methods

B:G4Cerenkov_modifiedTest
    cks standalone geant4 running using curand precooked randoms 
    cks ; ./G4Cerenkov_modifiedTest.sh 
    now using override_fNumPhotons to give shape (10000, 4, 4)

C:cks.py 
    cks python energy sampling implementation using same precooked randoms
    shape limited by the precooked randoms in use (10000, 4, 4)


"""
import numpy as np
import matplotlib.pyplot as plt 

from G4Cerenkov_modifiedTest import G4Cerenkov_modifiedTest as G4M
from cks import CKS  
from opticks.qudarap.tests.QCtxTest import QCtxTest  


if __name__ == '__main__':

     num = 10000
     a = QCtxTest.LoadCK(num)
     b = G4M.LoadGen()   
     c = CKS.Load() 

     a_loop = a.view(np.int32)[:,3,1]  
     b_loop = b.view(np.int32)[:,1,2]  
     c_loop = c.view(np.int64)[:,3,1]   

     assert np.all( a_loop == b_loop ) 
     assert np.all( c_loop == b_loop ) 

     ha = np.histogram( a_loop, a_loop.max()-1 )   

     fig, ax = plt.subplots()
     ax.plot( ha[1][:-1], ha[0], drawstyle="steps" ) 
     fig.show()


