#!/usr/bin/env python
"""

::

    cks
    ipython -i cputime.py 


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt 


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     fold = "/tmp/G4Cerenkov_modifiedTest"
     path = os.path.join(fold, "scan_GetAverageNumberOfPhotons.npy")
     figpath = os.path.join(fold,"scan_GetAverageNumberOfPhotons_cputime.png" )

     a = np.load(path)

     bi =  a[:,0]     
     t0 =  a[:,3]     
     t1 =  a[:,4]     


     title = [
              "examples/Geant4/CerenkovStandalone/cputime.py",
              "Comparison of GetAverageNumberOfPhotons CPU timings for various BetaInverse", 
              "the s2 integral approach yields fairly constant timings, only slower than asis when num_photons is zero",   
              path, 
             ]


     fig, ax = plt.subplots(figsize=[12.8, 7.2])

     fig.suptitle("\n".join(title))

     ax.plot( bi, t0, label="asis" )
     ax.plot( bi, t1, label="s2" )
     ax.set_ylim(0, 3)

     ax.set_xlabel("BetaInverse")
     ax.set_ylabel("CPU time (microseconds) ")
     ax.legend()

     fig.show()
     log.info("save to %s " % figpath)
     fig.savefig(figpath)


