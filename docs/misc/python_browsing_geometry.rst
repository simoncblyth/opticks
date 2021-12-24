python_browsing_geometry
==========================

Overview
---------

Several Opticks python scripts provide interactive investigation 
of a geometry model that is particularly usual during debugging. 
This page describes some of the more important scripts. 


GParts.py Preliminaries
-------------------------

Before using *GParts.py* its necessary to use the --savegparts option 
with any Opticks executable that does GGeo::deferredCreateGParts.

--savegparts
     save mm composite GParts to tmpdir (as this happens postcache it is not appropiate to save in geocache) 
     after deferred creation in GGeo::deferredCreateGParts

For example::

    OpSnapTest --savegparts    # writes $TMP/GParts/0,1,2,...


Q: is this still needed ?  Now that have the CSGFoundry geometry created in CSG_GGeo (cg) ?

A: that is a definite good point, as the GGeo::Load done by the cg conversion runs GGeo::deferredCreateGParts
   so could automate the saving of GParts into the CSGFoundry together with the conversion ?
   But perhaps could directly do much the same from the CSG geometry anyhow ?

TODO: standardize doing this as part of CSG_GGeo conversion, and move GParts into CSGFoundry directory  

* transitionally need both routes to work in anycase


GParts.py loads persisted analytic geometry into a simple python numpy array model
----------------------------------------------------------------------------------------

0. get overview of the "Solids" (in Opticks sense of compounded shapes aka GMergedMesh) with GParts.py::

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


ggeo.py 
---------


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



GNodeLib.py : Find the CE of some Chimney volumes 
---------------------------------------------------

Want to find the current nidx corresponding to some old logging::

    2019-04-21 00:27:12.438 FATAL [107202] [OpticksAim::setTarget@121] OpticksAim::setTarget  based on CenterExtent from m_mesh0  target 352851 aim 1 ce 0.0000,0.0000,19785.0000,1965.0000


With the below find that the nidx is now : 304632::


    CE
    array([    0.,     0., 19785.,  1965.], dtype=float32)


::

    epsilon:ana blyth$ GNodeLib.py --ulv --sli 0:None 
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.ulv found 131 unique LV names
    GLb1.bt02_HBeam0x34c1e00
    GLb1.bt05_HBeam0x34cf620
    GLb1.bt06_HBeam0x34d1e20
    GLb1.bt07_HBeam0x34d4620
    ..

    epsilon:ana blyth$ GNodeLib.py --ulv --sli 0:None  | grep Chimney 
    lLowerChimney0x4ee4270
    lLowerChimneyAcrylic0x4ee4490
    lLowerChimneyLS0x4ee46a0
    lLowerChimneySteel0x4ee48c0
    lUpperChimney0x4ee1f50
    lUpperChimneyLS0x4ee2050
    lUpperChimneySteel0x4ee2160
    lUpperChimneyTyvek0x4ee2270
    epsilon:ana blyth$ 

    epsilon:ana blyth$ GNodeLib.py --lv lLowerChimney0x4ee4270
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.lv:lLowerChimney0x4ee4270 matched 1 nodes 
    slice 0:10:1 
    [304632]
    ### Node idx:304632 

    TR
    array([[    1.,     0.,     0.,     0.],
           [    0.,     1.,     0.,     0.],
           [    0.,     0.,     1.,     0.],
           [    0.,     0., 19785.,     1.]], dtype=float32)

    BB
    array([[ -520.,  -520., 17820.,     1.],
           [  520.,   520., 21750.,     1.]], dtype=float32)

    ID
    array([ 304632,    3080, 7733270,       0], dtype=uint32)

    NI
    array([    96,     50, 304632,  67841], dtype=uint32)

    CE
    array([    0.,     0., 19785.,  1965.], dtype=float32)

    PV
    b'lLowerChimney_phys0x4ee5e60'

    LV
    b'lLowerChimney0x4ee4270'

    epsilon:ana blyth$ 




