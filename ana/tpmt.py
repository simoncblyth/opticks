#!/usr/bin/env python
"""
tpmt.py : PmtInBox Opticks vs G4 History comparisons
==========================================================

Loads test events from Opticks and Geant4 and 
compares their bounce histories.

Create the events by running tpmt- bash functions.

The convention is adopted of using positive tags for Opticks 
and negative ones of the same magnitude for the corresponding 
Geant4 simulated event.


See Also
-----------

:doc:`tpmt_debug`
      simulation debugging notes to acheive Opticks Geant4 match

:doc:`tpmt_distrib`
      comparison of distributions  


Expected Output
------------------

The expected output from the test is shown below. 
History step abbreviation: 

* *TO* torch step 
* *BT* boundary transmit
* *BR* boundary reflect
* *SA* surface absorb
* *SD* surface detect
* *AB* bulk absorb
* *SC* bulk scatter

Material abbreviations:

* *MO* Mineral Oil
* *Py* Pyrex
* *Vm* Vacuum
* *OV* Opaque Vacuum


.. code-block:: py 

    delta:ana blyth$ i
    SQLITE3_DATABASE=/usr/local/env/nuwa/mocknuwa.db
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    IPython profile: g4opticks

    In [1]: run pmt_test.py 
     1.175 100.000
     0.377 100.002
                          4:PmtInBox   -4:PmtInBox           c2 
                     8cd         67948        68252             0.68  [3 ] TO BT SA
                     7cd         21648        21369             1.81  [3 ] TO BT SD
                    8ccd          4581         4539             0.19  [4 ] TO BT BT SA
                      4d          3794         3864             0.64  [2 ] TO AB
                     86d           640          617             0.42  [3 ] TO SC SA
                     4cd           444          427             0.33  [3 ] TO BT AB
                    4ccd           350          362             0.20  [4 ] TO BT BT AB
                     8bd           283          259             1.06  [3 ] TO BR SA
                    8c6d            81           84             0.05  [4 ] TO SC BT SA
                   86ccd            51           57             0.33  [5 ] TO BT BT SC SA
                  8cbbcd            36           53             3.25  [6 ] TO BT BR BR BT SA
                     46d            40           30             1.43  [3 ] TO SC AB
                    7c6d            20           28             1.33  [4 ] TO SC BT SD
                     4bd            28           21             1.00  [3 ] TO BR AB
                8cbc6ccd             9            3             0.00  [8 ] TO BT BT SC BT BR BT SA
                    866d             8            4             0.00  [4 ] TO SC SC SA
                   8cc6d             7            7             0.00  [5 ] TO SC BT BT SA
                    86bd             6            4             0.00  [4 ] TO BR SC SA
                    8b6d             3            6             0.00  [4 ] TO SC BR SA
              cbccbbbbcd             4            0             0.00  [10] TO BT BR BR BR BR BT BT BR BT
                              100000       100000         0.91 
                          4:PmtInBox   -4:PmtInBox           c2 
                     ee4         90040        90048             0.00  [3 ] MO Py Py
                    44e4          4931         4901             0.09  [4 ] MO Py MO MO
                      44          3794         3864             0.64  [2 ] MO MO
                     444           991          927             2.14  [3 ] MO MO MO
                    ee44           101          113             0.67  [4 ] MO MO Py Py
                   444e4            52           58             0.33  [5 ] MO Py MO MO MO
                  44eee4            40           54             2.09  [6 ] MO Py Py Py MO MO
                    4444            17           14             0.29  [4 ] MO MO MO MO
                   44e44             8            7             0.00  [5 ] MO MO Py MO MO
                44ee44e4             6            3             0.00  [8 ] MO Py MO MO Py Py MO MO
                444e44e4             5            0             0.00  [8 ] MO Py MO MO Py MO MO MO
              44e4eeeee4             4            0             0.00  [10] MO Py Py Py Py Py MO Py MO MO
                  ee44e4             0            4             0.00  [6 ] MO Py MO MO Py Py
                   ee444             2            0             0.00  [5 ] MO MO MO Py Py
              44edbe44e4             2            0             0.00  [10] MO Py MO MO Py OV Vm Py MO MO
                  4444e4             0            2             0.00  [6 ] MO Py MO MO MO MO
              4ebdbe44e4             0            1             0.00  [10] MO Py MO MO Py OV Vm OV Py MO
              4e5dbe44e4             0            1             0.00  [10] MO Py MO MO Py OV Vm Bk Py MO
              eebdbe44e4             1            0             0.00  [10] MO Py MO MO Py OV Vm OV Py Py
                 44ee444             1            0             0.00  [7 ] MO MO MO Py Py MO MO
                              100000       100000         0.78 





"""
import os, sys, logging, argparse, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import vnorm
from opticks.ana.evt  import Evt



if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    args = opticks_main(doc=__doc__, tag="10", src="torch", det="PmtInBox", c2max=2.0, tagoffset=0)

    log.info("tag %s src %s det %s c2max %s  " % (args.utag,args.src,args.det, args.c2max))


    #seqs = ["TO BT BR BT BT BT BT SA"] 
    #seqs = ["TO BT BR BR BT SA"]
    seqs=[]

    try:
        a = Evt(tag="%s" % args.utag, src=args.src, det=args.det, seqs=seqs)
        b = Evt(tag="-%s" % args.utag , src=args.src, det=args.det, seqs=seqs)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

  

    log.info( " a : %s " % a.brief)
    log.info( " b : %s " % b.brief )

if 0:
    if a.valid:
        a0 = a.rpost_(0)
        #a0r = np.linalg.norm(a0[:,:2],2,1)
        a0r = vnorm(a0[:,:2])
        if len(a0r)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (a0r.min(),a0r.max())))

    if b.valid:
        b0 = b.rpost_(0)
        #b0r = np.linalg.norm(b0[:,:2],2,1)
        b0r = vnorm(b0[:,:2])
        if len(b0r)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (b0r.min(),b0r.max())))

if 1:
    Evt.compare_table(a,b, "seqhis_ana seqmat_ana".split(), lmx=20, c2max=args.c2max, cf=False)
    Evt.compare_table(a,b, "seqhis_ana seqmat_ana".split(), lmx=20, c2max=args.c2max, cf=True)
    Evt.compare_table(a,b, "pflags_ana hflags_ana".split(), lmx=20, c2max=None )

    #a.history_table()
    #b.history_table()


