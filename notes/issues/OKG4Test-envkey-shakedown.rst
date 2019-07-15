OKG4Test-envkey-shakedown
===============================



::

    gdb --args OKG4Test --envkey --xanalytic --geocenter

    OpticksProfile=ERROR gdb --args OKG4Test --envkey --xanalytic --geocenter --save



Lots of issues : need aligned+masked running for the shakedown
-------------------------------------------------------------------

* For a review of masked running see :doc:`where_mask_running` 

* upshot from that is : without lots of development are limited to input photons 

* BUT currently input photons only work for test geometry using NEmitPhotonsNPY

* how to make input photons work with full geometry ? 


Torch config input photons ?
-------------------------------------





Added code to find direct mode gdmlpath
--------------------------------------------

* by fishing in the geocache creating argline 

::

    Opticks::loadOriginCacheMeta
    Opticks::ExtractCacheMetaGDMLPath



Removed MPT assert
---------------------

::

    26 0x000000000040399a in main (argc=4, argv=0x7fffffffda28) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) c
    Continuing.
    2019-07-14 22:09:09.083 FATAL [122592] [CGDMLDetector::addMPTLegacyGDML@180]  UNEXPECTED TO SEE ONLY SOME Geant4 MATERIALS WITHOUT MPT  nmat 17 nmat_without_mpt 7
    OKG4Test: /home/blyth/opticks/cfg4/CGDMLDetector.cc:185: void CGDMLDetector::addMPTLegacyGDML(): Assertion `0' failed.
    
    Program received signal SIGABRT, Aborted.
    0x00007fffe1fe9207 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007fffe1fe9207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe1fea8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe1fe2026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe1fe20d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff4c839cb in CGDMLDetector::addMPTLegacyGDML (this=0x5d48920) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:185
    #5  0x00007ffff4c833ba in CGDMLDetector::init (this=0x5d48920) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:77
    #6  0x00007ffff4c83012 in CGDMLDetector::CGDMLDetector (this=0x5d48920, hub=0x6b8680, query=0x6c5280, sd=0x5d462c0) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:43
    #7  0x00007ffff4c2a384 in CGeometry::init (this=0x5d48870) at /home/blyth/opticks/cfg4/CGeometry.cc:80
    #8  0x00007ffff4c2a196 in CGeometry::CGeometry (this=0x5d48870, hub=0x6b8680, sd=0x5d462c0) at /home/blyth/opticks/cfg4/CGeometry.cc:63
    #9  0x00007ffff4c9bbce in CG4::CG4 (this=0x5abffb0, hub=0x6b8680) at /home/blyth/opticks/cfg4/CG4.cc:129
    #10 0x00007ffff7bd44c8 in OKG4Mgr::OKG4Mgr (this=0x7fffffffd6e0, argc=4, argv=0x7fffffffda28) at /home/blyth/opticks/okg4/OKG4Mgr.cc:87
    #11 0x000000000040399a in main (argc=4, argv=0x7fffffffda28) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) f 6
    #6  0x00007ffff4c83012 in CGDMLDetector::CGDMLDetector (this=0x5d48920, hub=0x6b8680, query=0x6c5280, sd=0x5d462c0) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:43
    43      init();
    (gdb) f 5
    #5  0x00007ffff4c833ba in CGDMLDetector::init (this=0x5d48920) at /home/blyth/opticks/cfg4/CGDMLDetector.cc:77
    77      addMPTLegacyGDML(); 
    (gdb) quit




Slow CloseGeometry
------------------------

* dont recall such a slow G4RunManager::RunInitialization previously 


Huh, long time voxeling::

    gdb) bt
    #0  0x00007ffff3731b9d in std::__uninitialized_copy<false>::__uninit_copy<std::move_iterator<HepGeom::Plane3D<double>*>, HepGeom::Plane3D<double>*> (__first=..., __last=..., __result=0x304048f0) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:76
    #1  0x00007ffff3731a62 in std::uninitialized_copy<std::move_iterator<HepGeom::Plane3D<double>*>, HepGeom::Plane3D<double>*> (__first=..., __last=..., __result=0x304048f0) at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:117
    #2  0x00007ffff373180c in std::__uninitialized_copy_a<std::move_iterator<HepGeom::Plane3D<double>*>, HepGeom::Plane3D<double>*, HepGeom::Plane3D<double> > (__first=..., __last=..., __result=0x304048f0)
        at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:258
    #3  0x00007ffff37312e0 in std::__uninitialized_move_if_noexcept_a<HepGeom::Plane3D<double>*, HepGeom::Plane3D<double>*, std::allocator<HepGeom::Plane3D<double> > > (__first=0x30400020, __last=0x304000a0, __result=0x304048f0, __alloc=...)
        at /usr/include/c++/4.8.2/bits/stl_uninitialized.h:281
    #4  0x00007fffed973366 in std::vector<HepGeom::Plane3D<double>, std::allocator<HepGeom::Plane3D<double> > >::_M_emplace_back_aux<HepGeom::Plane3D<double> >(HepGeom::Plane3D<double>&&) (this=0x7fffffffa430)
        at /usr/include/c++/4.8.2/bits/vector.tcc:412
    #5  0x00007fffed9727c9 in std::vector<HepGeom::Plane3D<double>, std::allocator<HepGeom::Plane3D<double> > >::emplace_back<HepGeom::Plane3D<double> >(HepGeom::Plane3D<double>&&) (this=0x7fffffffa430) at /usr/include/c++/4.8.2/bits/vector.tcc:101
    #6  0x00007fffed971658 in std::vector<HepGeom::Plane3D<double>, std::allocator<HepGeom::Plane3D<double> > >::push_back(HepGeom::Plane3D<double>&&) (this=0x7fffffffa430, 
        __x=<unknown type in /home/blyth/local/opticks/lib/../externals/lib64/libG4geometry.so, CU 0x281cd2, DIE 0x29c783>) at /usr/include/c++/4.8.2/bits/stl_vector.h:920
    #7  0x00007fffed96ba26 in G4BoundingEnvelope::CreateListOfPlanes (this=0x7fffffffa810, baseA=std::vector of length 6, capacity 6 = {...}, baseB=std::vector of length 6, capacity 6 = {...}, pPlanes=std::vector of length 4, capacity 4 = {...})
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4BoundingEnvelope.cc:790
    #8  0x00007fffed96960d in G4BoundingEnvelope::CalculateExtent (this=0x7fffffffa810, pAxis=kYAxis, pVoxelLimits=..., pTransform3D=..., pMin=@0x7fffffffaac8: 8.9999999999999999e+99, pMax=@0x7fffffffaac0: -8.9999999999999999e+99)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4BoundingEnvelope.cc:547
    #9  0x00007fffeda91de0 in G4Polycone::CalculateExtent (this=0x178e9e40, pAxis=kYAxis, pVoxelLimit=..., pTransform=..., pMin=@0x7fffffffb0d8: 8.9999999999999999e+99, pMax=@0x7fffffffb0d0: -8.9999999999999999e+99)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/solids/specific/src/G4Polycone.cc:695
    #10 0x00007fffed99361e in G4SmartVoxelHeader::BuildNodes (this=0x304014c0, pVolume=0x1790bc80, pLimits=..., pCandidates=0x303e9100, pAxis=kYAxis)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:852
    #11 0x00007fffed99275f in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0x304014c0, pVolume=0x1790bc80, pLimits=..., pCandidates=0x303e9100)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:476
    #12 0x00007fffed9917fc in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0x304014c0, pVolume=0x1790bc80, pLimits=..., pCandidates=0x303e9100, pSlice=565)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:119
    #13 0x00007fffed99437c in G4SmartVoxelHeader::RefineNodes (this=0x303cdaa0, pVolume=0x1790bc80, pLimits=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:1244
    #14 0x00007fffed992ae7 in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0x303cdaa0, pVolume=0x1790bc80, pLimits=..., pCandidates=0x2cd991c0)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:568
    #15 0x00007fffed9917fc in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0x303cdaa0, pVolume=0x1790bc80, pLimits=..., pCandidates=0x2cd991c0, pSlice=213)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:119
    #16 0x00007fffed99437c in G4SmartVoxelHeader::RefineNodes (this=0x2cbae120, pVolume=0x1790bc80, pLimits=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:1244
    #17 0x00007fffed992ae7 in G4SmartVoxelHeader::BuildVoxelsWithinLimits (this=0x2cbae120, pVolume=0x1790bc80, pLimits=..., pCandidates=0x7fffffffbb20)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:568
    #18 0x00007fffed991cdb in G4SmartVoxelHeader::BuildVoxels (this=0x2cbae120, pVolume=0x1790bc80) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:258
    #19 0x00007fffed99170d in G4SmartVoxelHeader::G4SmartVoxelHeader (this=0x2cbae120, pVolume=0x1790bc80, pSlice=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4SmartVoxelHeader.cc:82
    #20 0x00007fffed97ed2d in G4GeometryManager::BuildOptimisations (this=0x2c7de520, allOpts=true, verbose=false) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4GeometryManager.cc:200
    #21 0x00007fffed97eaa5 in G4GeometryManager::CloseGeometry (this=0x2c7de520, pOptimise=true, verbose=false, pVolume=0x0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/geometry/management/src/G4GeometryManager.cc:102
    #22 0x00007ffff156c589 in G4RunManagerKernel::ResetNavigator (this=0x6cdcf0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:757
    #23 0x00007ffff156c3a6 in G4RunManagerKernel::RunInitialization (this=0x6cdcf0, fakeRun=false) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManagerKernel.cc:699
    #24 0x00007ffff155cf69 in G4RunManager::RunInitialization (this=0x6cdbd0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:313
    #25 0x00007ffff155cd0f in G4RunManager::BeamOn (this=0x6cdbd0, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:272
    #26 0x00007ffff4c9d7e8 in CG4::propagate (this=0x5abffb0) at /home/blyth/opticks/cfg4/CG4.cc:345
    #27 0x00007ffff7bd49d1 in OKG4Mgr::propagate_ (this=0x7fffffffd6e0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:201
    #28 0x00007ffff7bd487f in OKG4Mgr::propagate (this=0x7fffffffd6e0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:138
    #29 0x00000000004039a9 in main (argc=4, argv=0x7fffffffda28) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 



::

    g4-cls G4GeometryManager
    g4-cls G4SmartVoxelHeader







Saves inside geocache::




* unset OPTICKS_EVENT_BASE (it was set to $TMP avoids the need to set it like below)

::

    [blyth@localhost ana]$ OPTICKS_EVENT_BASE=/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/source tokg4.py
    args: /home/blyth/opticks/ana/tokg4.py
    [2019-07-14 23:09:49,502] p221393 {<module>            :tokg4.py  :25} INFO     - tag 1 src torch det g4live c2max [1.5, 2.0, 2.5]  
    [2019-07-14 23:09:49,503] p221393 {__init__            :evt.py    :173} INFO     - [ ? 
    [2019-07-14 23:09:49,582] p221393 {__init__            :evt.py    :233} INFO     - ] ? 
    [2019-07-14 23:09:49,582] p221393 {__init__            :evt.py    :173} INFO     - [ ? 
    [2019-07-14 23:09:49,620] p221393 {__init__            :evt.py    :233} INFO     - ] ? 
    [2019-07-14 23:09:49,620] p221393 {<module>            :tokg4.py  :37} INFO     -  a : ./g4live/torch/  1 :  20190714-2255 maxbounce:9 maxrec:10 maxrng:3000000 /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/source/./evt/g4live/torch/1/fdom.npy () 
    [2019-07-14 23:09:49,620] p221393 {<module>            :tokg4.py  :38} INFO     -  b : ./g4live/torch/ -1 :  20190714-2255 maxbounce:9 maxrec:10 maxrng:3000000 /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/source/./evt/g4live/torch/-1/fdom.npy (recstp) 
    A Evt(  1,"torch","g4live",pfx=".", seqs="[]", msli="0:100k:" ) 20190714-2255 
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/source/./evt/g4live/torch/1
     file_photons 10k   load_slice 0:100k:   loaded_photons 10k 
       fdom :            - :        3,1,4 : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
       idom :            - :        1,1,4 : (metadata) maxbounce:9 maxrec:10 maxrng:3000000  
         gs :            - :        1,6,4 : (gensteps) 
         ox :      10k,4,4 :      10k,4,4 : (photons) final photon step   
         wl :            - :          10k : (photons) wavelength 
       post :            - :        10k,4 : (photons) final photon step: position, time 
       dirw :            - :        10k,4 : (photons) final photon step: direction, weight  
       polw :            - :        10k,4 : (photons) final photon step: polarization, wavelength  
     pflags :            - :          10k : (photons) final photon step: flags  
         c4 :            - :          10k : (photons) final photon step: dtype split uint8 view of ox flags 
         ht :            - :     2385,4,4 : (hits) surface detect SD final photon steps 
        hwl :            - :         2385 : (hits) wavelength 
      hpost :            - :       2385,4 : (hits) final photon step: position, time 
      hdirw :            - :       2385,4 : (hits) final photon step: direction, weight  
      hpolw :            - :       2385,4 : (hits) final photon step: polarization, wavelength  
     hflags :            - :         2385 : (hits) final photon step: flags  
        hc4 :            - :         2385 : (hits) final photon step: dtype split uint8 view of ox flags 
         rx :   10k,10,2,4 :   10k,10,2,4 : (records) photon step records 
         ph :      10k,1,2 :      10k,1,2 : (records) photon history flag/material sequence 
         so :            - :              : (source) input CPU side emitconfig photons, or initial cerenkov/scintillation 
    B Evt( -1,"torch","g4live",pfx=".", seqs="[]", msli="0:100k:" ) 20190714-2255 
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/source/./evt/g4live/torch/-1
     file_photons 10k   load_slice 0:100k:   loaded_photons 10k 
       fdom :            - :        3,1,4 : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
       idom :            - :        1,1,4 : (metadata) maxbounce:9 maxrec:10 maxrng:3000000  
         gs :            - :        1,6,4 : (gensteps) 
         ox :      10k,4,4 :      10k,4,4 : (photons) final photon step   
         wl :            - :          10k : (photons) wavelength 
       post :            - :        10k,4 : (photons) final photon step: position, time 
       dirw :            - :        10k,4 : (photons) final photon step: direction, weight  
       polw :            - :        10k,4 : (photons) final photon step: polarization, wavelength  
     pflags :            - :          10k : (photons) final photon step: flags  
         c4 :            - :          10k : (photons) final photon step: dtype split uint8 view of ox flags 
         ht :            - :     1152,4,4 : (hits) surface detect SD final photon steps 
        hwl :            - :         1152 : (hits) wavelength 
      hpost :            - :       1152,4 : (hits) final photon step: position, time 
      hdirw :            - :       1152,4 : (hits) final photon step: direction, weight  
      hpolw :            - :       1152,4 : (hits) final photon step: polarization, wavelength  
     hflags :            - :         1152 : (hits) final photon step: flags  
        hc4 :            - :         1152 : (hits) final photon step: dtype split uint8 view of ox flags 
         rx :   10k,10,2,4 :   10k,10,2,4 : (records) photon step records 
         ph :      10k,1,2 :      10k,1,2 : (records) photon history flag/material sequence 
         so :            - :              : (source) input CPU side emitconfig photons, or initial cerenkov/scintillation 
    [blyth@localhost ana]$ 


::

    [blyth@localhost opticks]$ echo $TMP
    /home/blyth/local/opticks/tmp
    [blyth@localhost opticks]$ echo $OPTICKS_EVENT_BASE
    /home/blyth/local/opticks/tmp

    [blyth@localhost opticks]$ unset TMP ; unset OPTICKS_EVENT_BASE
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ tokg4.py 




