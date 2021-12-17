making_flightpath_raytrace_movies
===================================


0. get overview of the "Solids" (in Opticks sense of compounded shapes aka GMergedMesh) with GParts.py::

    OpSnapTest --savegparts    
    # any Opticks executable can do this (necessary as GParts are now postcache so this does not belong in geocache)
    # the parts are saved into $TMP/GParts

    # TODO: could standardize doing as part of CSG_GGeo conversion 


    epsilon:ana blyth$ GParts.py 
    Solid 0 : /tmp/blyth/opticks/GParts/0 : primbuf (3084, 4) partbuf (17346, 4, 4) tranbuf (7917, 3, 4, 4) idxbuf (3084, 4) 
    Solid 1 : /tmp/blyth/opticks/GParts/1 : primbuf (5, 4) partbuf (7, 4, 4) tranbuf (5, 3, 4, 4) idxbuf (5, 4) 
    Solid 2 : /tmp/blyth/opticks/GParts/2 : primbuf (6, 4) partbuf (30, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 
    Solid 3 : /tmp/blyth/opticks/GParts/3 : primbuf (6, 4) partbuf (54, 4, 4) tranbuf (29, 3, 4, 4) idxbuf (6, 4) 
    Solid 4 : /tmp/blyth/opticks/GParts/4 : primbuf (6, 4) partbuf (28, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 
    Solid 5 : /tmp/blyth/opticks/GParts/5 : primbuf (1, 4) partbuf (3, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
    Solid 6 : /tmp/blyth/opticks/GParts/6 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (9, 3, 4, 4) idxbuf (1, 4) 
    Solid 7 : /tmp/blyth/opticks/GParts/7 : primbuf (1, 4) partbuf (1, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
    Solid 8 : /tmp/blyth/opticks/GParts/8 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (11, 3, 4, 4) idxbuf (1, 4) 
    Solid 9 : /tmp/blyth/opticks/GParts/9 : primbuf (130, 4) partbuf (130, 4, 4) tranbuf (130, 3, 4, 4) idxbuf (130, 4) 

    * in above 6 and 8 look interesting : they are single prim with 31 parts(aka nodes) 
      that is a depth 5 tree and potential performance problem


1. use ggeo.py and triplet indexing to find the corresponding global node index (nidx)::

    epsilon:opticks blyth$ ggeo.py 5:9/0/* --names
    nrpo(  69668     5     0     0 )                                     lUpper_phys0x35b5ac0                                          lUpper0x35b5a00 
    nrpo(  69078     6     0     0 )                                 lFasteners_phys0x34ce040                                      lFasteners0x34cdf00 
    nrpo(  68488     7     0     0 )                                     lSteel_phys0x352c890                                          lSteel0x352c760 
    nrpo(  70258     8     0     0 )                                  lAddition_phys0x35ff770                                       lAddition0x35ff5f0 

    epsilon:ana blyth$ ggeo.py 5:9/0/*  --brief
    nidx: 69668 triplet: 5000000 sh:5f0014 sidx:    0   nrpo(  69668     5     0     0 )  shape(  95  20                       base_steel0x360d8f0                            Water///Steel) 
    nidx: 69078 triplet: 6000000 sh:5e0014 sidx:    0   nrpo(  69078     6     0     0 )  shape(  94  20                             uni10x34cdcb0                            Water///Steel) 
    nidx: 68488 triplet: 7000000 sh:5d0014 sidx:    0   nrpo(  68488     7     0     0 )  shape(  93  20                   sStrutBallhead0x352a360                            Water///Steel) 
    nidx: 70258 triplet: 8000000 sh:600010 sidx:    0   nrpo(  70258     8     0     0 )  shape(  96  16                     uni_acrylic30x35ff3d0                          Water///Acrylic) 

2. create an eye-look-up flight path, that is saved to /tmp/flightpath.npy::

   flight.sh --roundaboutxy 

3. launch visualization, press U to switch to the animated InterpolatedView created from the flightpath::

   OTracerTest --targetpvn lFasteners_phys
   OTracerTest --target 69078

4. for non-interative raytrace jpg snaps around the flightpath::

   PERIOD=8 PVN=lLowerChimney_phys EMM=~5, flight.sh --rtx 1 --cvd 1 

5. make an mp4 from the jpg snaps.  flight.sh can automatically 
   create the mp4 assuming env repo and ffmpeg are installed

6. when doing the above snaps on remote ssh node P::

   okop ; cd tests
   ./OpFlightPathTest.sh grab 

