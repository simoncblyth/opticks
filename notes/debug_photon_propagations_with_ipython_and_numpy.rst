debug_photon_propagations_with_ipython_and_numpy
===================================================


::

    epsilon:~ blyth$ cd ~/opticks/ana/tests
    epsilon:tests blyth$ ./check.sh 
    CSGFoundry.CFBase returning [/Users/blyth/.opticks/ntds3/G4CXOpticks], note:[via CFBASE] 
    Fold : symbol f base /Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimulateTest/ALL 
    f

    CMDLINE:/Users/blyth/opticks/ana/tests/check.py
    f.base:/Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimulateTest/ALL

      : f.sframe_meta                                      :                    2 : 71 days, 21:09:58.832379 
      : f.genstep                                          :            (1, 6, 4) : 71 days, 21:10:04.233474 
      : f.hit                                              :          (948, 4, 4) : 71 days, 21:10:03.571458 
      : f.seq                                              :            (1000, 2) : 71 days, 21:09:58.833446 
      : f.record_meta                                      :                    1 : 71 days, 21:09:58.834056 
      : f.rec_meta                                         :                    1 : 71 days, 21:10:00.071103 
      : f.rec                                              :     (1000, 10, 2, 4) : 71 days, 21:10:00.071581 
      : f.NPFold_meta                                      :                    8 : 71 days, 21:10:06.361451 
      : f.record                                           :     (1000, 10, 4, 4) : 71 days, 21:09:58.834939 
      : f.sframe                                           :            (4, 4, 4) : 71 days, 21:09:58.832854 
      : f.flat                                             :           (1000, 64) : 71 days, 21:10:04.234338 
      : f.NPFold_index                                     :                    9 : 71 days, 21:10:06.362230 
      : f.prd                                              :     (1000, 10, 2, 4) : 71 days, 21:10:00.483583 
      : f.photon                                           :         (1000, 4, 4) : 71 days, 21:10:02.727718 
      : f.tag                                              :            (1000, 4) : 71 days, 21:09:58.831017 

     min_stamp : 2022-08-24 12:18:28.467579 
     max_stamp : 2022-08-24 12:18:35.998792 
     dif_stamp : 0:00:07.531213 
     age_stamp : 71 days, 21:09:58.831017 

    In [1]: seqhis_(f.seq[:10,0])      # dumping photon histories for first 10 
    Out[1]: 
    ['TO BT BT BT BT SD',
     'TO BT BT BT BT SD',
     'TO BT BT BT BT SD',
     'TO BT BT BT BT SD',
     'TO BT BT BT BT SD',
     'TO AB',
     'TO BT BT BT BT SD',
     'TO BT BT BT BT SD',
     'TO BT BT BT BT SD',
     'TO BT BT BT BT SD']

Checking the boundary and prim name at which the photon ended::

    In [2]: cf_bnd_(f.photon[0])
    Out[2]: 'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum'

    In [4]: cf_prim_(f.photon[0])
    Out[4]: 'NNVTMCPPMT_PMT_20inch_inner1_solid_head'

Checking the prim name for the record step points of the photon::

    In [6]: cf_prim_(f.record[0,1])
    Out[6]: 'NNVTMCPPMTsMask_virtual'

    In [7]: cf_prim_(f.record[0,2])
    Out[7]: 'NNVTMCPPMTsMask'

    In [8]: cf_prim_(f.record[0,3]) 
    Out[8]: 'NNVTMCPPMTsMask'

    In [9]: cf_prim_(f.record[0,4])
    Out[9]: 'NNVTMCPPMT_PMT_20inch_pmt_solid_head'

    In [10]: cf_prim_(f.record[0,5])
    Out[10]: 'NNVTMCPPMT_PMT_20inch_inner1_solid_head'

    In [11]: cf_prim_(f.record[0,6])   ## beyond the end of photon step points, just get zeros
    Out[11]: 'sWorld'


Checking the boundary for the record step points of the photon::

    In [3]: cf_bnd_(f.record[0,0])
    Out[3]: 'Galactic///Galactic'

    In [4]: cf_bnd_(f.record[0,1])
    Out[4]: 'Water///Water'

    In [5]: cf_bnd_(f.record[0,2])
    Out[5]: 'Water///AcrylicMask'

    In [6]: cf_bnd_(f.record[0,3])
    Out[6]: 'Water///AcrylicMask'

    In [7]: cf_bnd_(f.record[0,4])
    Out[7]: 'Water///Pyrex'

    In [8]: cf_bnd_(f.record[0,5])
    Out[8]: 'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum'

    In [9]: cf_bnd_(f.record[0,6])
    Out[9]: 'Galactic///Galactic'






For the above to work you will need to modify envvars in check.sh::

    epsilon:tests blyth$ cat check.sh 
    #!/bin/bash -l 

    export CFBASE=$HOME/.opticks/ntds3/G4CXOpticks
    export FOLD=$CFBASE/G4CXSimulateTest/ALL

    ${IPYTHON:-ipython} --pdb -i check.py 


CFBASE
    directory that contains the persisted CSGFoundry directory 
FOLD
    directory containing SEvt arrays to examine
   

To persist the geometry use, for example::

    export G4CXOpticks__setGeometry_saveGeometry=$HOME/.opticks/GEOM/example_pet 

See what that does by looking at g4cx/G4CXOpticks.cc



matplotlib and pyvista
-------------------------

matplotlib is a very popular python/NumPy plotting package that 
is used extensively for 2D plotting by Opticks scripts. 
However matplotlib has 3D plotting performance 
so bad that its not worth using for 3D. 

pyvista provides a convenient interface to the VTK : Visualization Toolkit. 
which features 3D plotting of large data sets with GPU acceleration.

You can install pyvista and matplotlib using anaconda

Note however that pyvista is a very fast moving project and opticks
plotting scripts do not work with the latest pyvista due 
to difficulties with updating VTK on my ancient laptop. 
So if you want to use Opticks pyvista plotting you will need
to install an older pyvista and the corresponding VTK.:: 

    In [1]: import pyvista as pv 
    In [2]: pv.__version__
    Out[2]: '0.25.3'

You are however free to install a newer pyvista and VTK
but that means you will have to rewrite all the pv 
plotting machinery. If you do that please share them. 

If you find import errors from lack of pyvista or matplotlib  
please report them as the intention is for the basics of the analysis
machinery to work without such packages. 
Of course you will not be able to make 2D or 3D plots 
without those packages. 






