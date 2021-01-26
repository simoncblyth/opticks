G4OKTest_fail_from_nhit_nhiy_mismatch
======================================


::

    2021-01-25 23:31:28.398 INFO  [4947927] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2021-01-25 23:31:28.414 INFO  [4947927] [OEvent::downloadHits@443]  nhit 17 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2021-01-25 23:31:28.417 INFO  [4947927] [OEvent::downloadHiys@476]  nhiy 47 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    Assertion failed: (nhit == nhiy), function download, file /Users/blyth/opticks/optixrap/OEvent.cc, line 511.
    Abort trap: 6
    epsilon:opticks blyth$ 



Commenting okc/OpticksSwitches.h:WITH_WAY_BUFFER get::

    2021-01-26 10:23:13.223 INFO  [5129070] [GPho::dump@367]  selectionName All opt nidx,nrpo,post,lpst,okfl count 18
    2021-01-26 10:23:13.223 INFO  [5129070] [G4OKTest::checkHits@337]  eventID 2 num_hit 18
        0 boundary   -2 sensorIndex    90 nodeIndex    90 photonIndex   276 flag_mask          5040 sensor_identifier         5a wavelength      430 time  346.507 SD|TO|EX
        1 boundary   -2 sensorIndex    88 nodeIndex    88 photonIndex   347 flag_mask          5040 sensor_identifier         58 wavelength      430 time  318.201 SD|TO|EX
        2 boundary   -2 sensorIndex   153 nodeIndex   153 photonIndex   364 flag_mask          5040 sensor_identifier         99 wavelength      430 time  324.493 SD|TO|EX
        3 boundary   -2 sensorIndex    84 nodeIndex    84 photonIndex   417 flag_mask          5040 sensor_identifier         54 wavelength      430 time  599.347 SD|TO|EX
        4 boundary   -2 sensorIndex    90 nodeIndex    90 photonIndex   425 flag_mask          5040 sensor_identifier         5a wavelength      430 time  353.704 SD|TO|EX
        5 boundary   -2 sensorIndex   104 nodeIndex   104 photonIndex   606 flag_mask          5040 sensor_identifier         68 wavelength      430 time  307.724 SD|TO|EX
        6 boundary   -2 sensorIndex   170 nodeIndex   170 photonIndex   622 flag_mask          5040 sensor_identifier         aa wavelength      430 time  351.877 SD|TO|EX
        7 boundary   -2 sensorIndex   104 nodeIndex   104 photonIndex  1154 flag_mask          5040 sensor_identifier         68 wavelength      430 time  308.367 SD|TO|EX
        8 boundary   -2 sensorIndex    87 nodeIndex    87 photonIndex  1224 flag_mask          5040 sensor_identifier         57 wavelength      430 time  355.064 SD|TO|EX
        9 boundary   -2 sensorIndex    91 nodeIndex    91 photonIndex  1378 flag_mask          9040 sensor_identifier         5b wavelength      430 time  422.102 SD|TO|EC
       10 boundary   -2 sensorIndex   103 nodeIndex   103 photonIndex  1547 flag_mask          5040 sensor_identifier         67 wavelength      430 time  353.437 SD|TO|EX
       11 boundary   -2 sensorIndex   219 nodeIndex   219 photonIndex  1748 flag_mask          5040 sensor_identifier         db wavelength      430 time   768.92 SD|TO|EX
       12 boundary   -2 sensorIndex   151 nodeIndex   151 photonIndex  1818 flag_mask          5040 sensor_identifier         97 wavelength      430 time  362.691 SD|TO|EX
       13 boundary   -2 sensorIndex   153 nodeIndex   153 photonIndex  2291 flag_mask          9040 sensor_identifier         99 wavelength      430 time  316.385 SD|TO|EC
       14 boundary   -2 sensorIndex   166 nodeIndex   166 photonIndex  2487 flag_mask          5040 sensor_identifier         a6 wavelength      430 time  422.753 SD|TO|EX
       15 boundary   -2 sensorIndex    89 nodeIndex    89 photonIndex  2539 flag_mask          5040 sensor_identifier         59 wavelength      430 time  308.097 SD|TO|EX
       16 boundary   -2 sensorIndex   100 nodeIndex   100 photonIndex  2635 flag_mask          5040 sensor_identifier         64 wavelength      430 time  589.116 SD|TO|EX
       17 boundary   -2 sensorIndex   232 nodeIndex   232 photonIndex  2741 flag_mask          5040 sensor_identifier         e8 wavelength      430 time  722.662 SD|TO|EX
    2021-01-26 10:23:13.223 DEBUG [5129070] [G4OKTest::propagate@330] ]
    epsilon:presentation blyth$ 


::

    cd ~/opticks

    export OPTICKS_EMBEDDED_COMMANDLINE="--dumphit --dumphiy"

    vi sysrap/PLOG_INIT.hh   # set FMT for easy log comparison 
    oo # rebuild : this is a big rebuild as practically all sources use this header

    G4OKTest > G4OKTest_with_way_buffer.log

    vi optickscore/OpticksSwitches.h   # comment WITH_WAY_BUFFER

    oo   # rebuild  : tedious to have to make this switch at compile time, as have to then rebuild the lot 

    G4OKTest > G4OKTest_without_way_buffer.log 



Actions
----------

* in sysrap/PLOG_INIT.hh changed logging format for easier log comparison



 
::

    epsilon:g4ok blyth$ OEvent=INFO G4OKTest 


1st event fine::

    OEvent::download@501: [
    OEvent::download@541: [ id 0
    OEvent::download@597: ]
    OEvent::downloadHitsCompute@636:  nhit 47 hit 47,4,4
    OEvent::downloadHits@443:  nhit 47 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@506:  nhit 47
    OEvent::downloadHiysCompute@653: [
    OEvent::downloadHiysCompute@670:  nhiy 47 hiy 47,2,4
    OEvent::downloadHiysCompute@681: ]
    OEvent::downloadHiys@478:  nhiy 47 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@510:  nhiy 47
    OEvent::download@523: ]

2nd and subsequnet the hiy is stuck at 47::

    OEvent::download@501: [
    OEvent::download@541: [ id 1
    OEvent::download@597: ]
    OEvent::downloadHitsCompute@636:  nhit 17 hit 17,4,4
    OEvent::downloadHits@443:  nhit 17 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@506:  nhit 17
    OEvent::downloadHiysCompute@653: [
    OEvent::downloadHiysCompute@670:  nhiy 47 hiy 47,2,4
    OEvent::downloadHiysCompute@681: ]
    OEvent::downloadHiys@478:  nhiy 47 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@510:  nhiy 47
    OEvent::download@514:  nhit 17 nhiy 47
    OEvent::download@523: ]


Added reset for hiy_buffer in optickscore/OpticksEvent::

    epsilon:optickscore blyth$ OpticksEvent=INFO OEvent=INFO G4OKTest 


Hmm not enough, 47 still stuck in the hy?::

    epsilon:optickscore blyth$ np.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/*.npy
    a :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/dg.npy :         (5000, 1, 4) : 99cb258a6605a5f7529f33c0cff52350 : 20210126-1417 
    b :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ox.npy :         (5000, 4, 4) : 8f516940d25682999e531f9f2edffc9a : 20210126-1417 
    c : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ox_local.npy :         (5000, 4, 4) : 9e7a40e1d3c1cc278a7e51b3ef39dcaa : 20210125-2325 
    d :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ph.npy :         (5000, 1, 2) : cd73b79709b4eabb0688578f0537eeb7 : 20210126-1417 
    e :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ps.npy :         (5000, 1, 4) : cdad8845616cc8df3ded3bda451d0628 : 20210126-1417 
    f :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/rs.npy :     (5000, 10, 1, 4) : 0e35047db9f17a66637e72a38ddbd320 : 20210126-1417 
    g :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/rx.npy :     (5000, 10, 2, 4) : 4b4c11d9b874c163f094a0c058854974 : 20210126-1417 
    h :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy :         (5000, 2, 4) : c7dea1c049275ed136a93259c00e74ab : 20210126-1417 
    i :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ht.npy :           (47, 4, 4) : 343a880d8ed631697428673f781cef6f : 20210126-1417 
    j :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/hy.npy :           (47, 2, 4) : f596f24014973772d0002fafce4a68df : 20210126-1417 
    k :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : 16b5b59871cef2abbfc9ba3499123d2d : 20210126-1417 
    l : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileAcc.npy :               (1, 4) : 78acdcbd8b75db33c249807a8c89ea49 : 20210126-1417 
    m : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileAccLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1417 
    n : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileLis.npy :                 (1,) : 611938dbf2d33d981f675a7ef2f60ea4 : 20210126-1417 
    o : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileLisLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1417 
    p :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : b1c03673018cd1e81a7f5080cdaf31e8 : 20210126-1417 
    q :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20210126-1417 
    r : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfile.npy :               (0, 4) : d1d127c8c0c63b61d6f6bf917e6b3d7b : 20210126-1417 
    s : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileLabels.npy :              (0, 64) : 4051d15b97452eb1de610547e562fe21 : 20210126-1417 
    epsilon:optickscore blyth$ 
    epsilon:optickscore blyth$ np.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/*.npy
    a :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/dg.npy :         (2000, 1, 4) : 600d5e2b539f1aff0534bd80df0bfe78 : 20210126-1417 
    b :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ox.npy :         (2000, 4, 4) : 49576da794c90190adabe710d18cb42a : 20210126-1417 
    c :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ph.npy :         (2000, 1, 2) : 0581e94d0a5297fe54aa03b9d90c3f71 : 20210126-1417 
    d :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ps.npy :         (2000, 1, 4) : fe06a58759f74b6eec2e4ee64552be4d : 20210126-1417 
    e :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/rs.npy :     (2000, 10, 1, 4) : 1404aa74a729f9b87463f1d5e2595428 : 20210126-1417 
    f :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/rx.npy :     (2000, 10, 2, 4) : e541c4e0be35bc1d9c5e3d4175a49eda : 20210126-1417 
    g :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/wy.npy :         (2000, 2, 4) : 5386979bff5c09f1b9dfdbf9fc244a5a : 20210126-1417 
    h :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/hy.npy :           (47, 2, 4) : f596f24014973772d0002fafce4a68df : 20210126-1417 
    i :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ht.npy :           (17, 4, 4) : cdf38607f9e1086a44e306dc49472197 : 20210126-1417 
    j :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/fdom.npy :            (3, 1, 4) : 16b5b59871cef2abbfc9ba3499123d2d : 20210126-1417 
    k : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileAcc.npy :               (1, 4) : 78acdcbd8b75db33c249807a8c89ea49 : 20210126-1417 
    l : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileAccLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1417 
    m : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileLis.npy :                 (1,) : 611938dbf2d33d981f675a7ef2f60ea4 : 20210126-1417 
    n : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileLisLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1417 
    o :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/gs.npy :            (1, 6, 4) : 3dd428490778dd3298816b8802d2d630 : 20210126-1417 
    p :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20210126-1417 
    q : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfile.npy :               (0, 4) : d1d127c8c0c63b61d6f6bf917e6b3d7b : 20210126-1417 
    r : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileLabels.npy :              (0, 64) : 4051d15b97452eb1de610547e562fe21 : 20210126-1417 
    epsilon:optickscore blyth$ 



Looks like are missing a reset or resize of the GPU side hiy buffer::

    OEvent::download@501: [
    OEvent::download@541: [ id 1
    OEvent::download@597: ]
    OEvent::downloadHitsCompute@623: into hit array :0,4,4
    OEvent::downloadHitsCompute@636:  nhit 17 hit 17,4,4
    OEvent::downloadHits@443:  nhit 17 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@506:  nhit 17
    OEvent::downloadHiysCompute@659: [
    OEvent::downloadHiysCompute@664: into hiy array :0,2,4
    OEvent::downloadHiysCompute@678:  nhiy 47 hiy 47,2,4
    OEvent::downloadHiysCompute@689: ]
    OEvent::downloadHiys@478:  nhiy 47 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@510:  nhiy 47
    OEvent::download@514:  nhit 17 nhiy 47
    OEvent::download@523: ]

Added resizing of the way buffer that the hiy buffer selectes from in OEvent::


    +#ifdef WITH_WAY_BUFFER
    +    NPY<float>* way = evt->getWayData() ; 
    +    assert(way);
    +    m_ocontext->resizeBuffer<float>(m_way_buffer,  way, "way");
    +#endif
    +


That fixes it::

    epsilon:optixrap blyth$ OpticksEvent=INFO OEvent=INFO G4OKTest      
    ...

    [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ 
    [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ 1
    [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ 

    GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@271:  target_lvname /dd/Geometry/AD/lvADE0xc2a78c00x3ef9140 nidxs.size() 2 nidx 3153
    G4OKTest::collectGensteps@301:  eventID 1 num_genstep_photons 2000
    G4OKTest::propagate@309: [
    OpticksEvent::resize@1251:  num_photons 2000 num_records 20000 maxrec 10 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2
    OpPropagator::propagate@73: evtId(1) OK COMPUTE DEVELOPMENT
    OEvent::upload@384: [ id 1
    OEvent::setEvent@54:  this (OEvent*) 0x7fd617191e80 evt (OpticksEvent*) 0x7fd6188e4fa0
    OEvent::resizeBuffers@323:  genstep 1,6,4 nopstep 0,4,4 photon 2000,4,4 debug 2000,1,4 way 2000,2,4 source NULL record 2000,10,2,4 phosel 2000,1,4 recsel 2000,10,1,4 sequence 2000,1,2 seed 2000,1,1 hit 0,4,4
    OEvent::uploadGensteps@420: (COMPUTE) id 1 1,6,4 -> 2000
    OEvent::upload@403: ] id 1
    OpSeeder::seedPhotonsFromGenstepsViaOptiX@174: SEEDING TO SEED BUF  
    OEvent::markDirty@250: 
    OPropagator::launch@268: LAUNCH NOW  --printenabled  printLaunchIndex ( 0 0 0) -
    OPropagator::launch@277: LAUNCH DONE
    OPropagator::launch@279: 0 : (0;2000,1) 
    BTimes::dump@183: OPropagator::launch
                    launch002                 0.337462
    OpIndexer::indexSequenceCompute@237: OpIndexer::indexSequenceCompute
    OEvent::download@525: [
    OEvent::download@565: [ id 1
    OEvent::download@621: ]
    OEvent::downloadHitsCompute@647: into hit array :0,4,4
    OEvent::downloadHitsCompute@660:  nhit 17 hit 17,4,4
    OEvent::downloadHits@467:  nhit 17 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@530:  nhit 17
    OEvent::downloadHiysCompute@683: [
    OEvent::downloadHiysCompute@688: into hiy array :0,2,4
    OEvent::downloadHiysCompute@702:  nhiy 17 hiy 17,2,4
    OEvent::downloadHiysCompute@713: ]
    OEvent::downloadHiys@502:  nhiy 17 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    OEvent::download@534:  nhiy 17
    OEvent::download@547: ]



::

    epsilon:optixrap blyth$  np.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/*.npy
    a :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/dg.npy :         (5000, 1, 4) : 99cb258a6605a5f7529f33c0cff52350 : 20210126-1501 
    b :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ox.npy :         (5000, 4, 4) : 8f516940d25682999e531f9f2edffc9a : 20210126-1501 
    c : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ox_local.npy :         (5000, 4, 4) : 9e7a40e1d3c1cc278a7e51b3ef39dcaa : 20210125-2325 
    d :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ph.npy :         (5000, 1, 2) : cd73b79709b4eabb0688578f0537eeb7 : 20210126-1501 
    e :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ps.npy :         (5000, 1, 4) : cdad8845616cc8df3ded3bda451d0628 : 20210126-1501 
    f :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/rs.npy :     (5000, 10, 1, 4) : 0e35047db9f17a66637e72a38ddbd320 : 20210126-1501 
    g :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/rx.npy :     (5000, 10, 2, 4) : 4b4c11d9b874c163f094a0c058854974 : 20210126-1501 
    h :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy :         (5000, 2, 4) : c7dea1c049275ed136a93259c00e74ab : 20210126-1501 
    i :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ht.npy :           (47, 4, 4) : 343a880d8ed631697428673f781cef6f : 20210126-1501 
    j :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/hy.npy :           (47, 2, 4) : f596f24014973772d0002fafce4a68df : 20210126-1501 
    k :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : 16b5b59871cef2abbfc9ba3499123d2d : 20210126-1501 
    l : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileAcc.npy :               (1, 4) : f949b607e29aa73b13f834396406217c : 20210126-1501 
    m : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileAccLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1501 
    n : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileLis.npy :                 (1,) : c72246e7e3772306d6e202419e22f6b0 : 20210126-1501 
    o : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileLisLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1501 
    p :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : b1c03673018cd1e81a7f5080cdaf31e8 : 20210126-1501 
    q :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20210126-1501 
    r : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfile.npy :               (0, 4) : d1d127c8c0c63b61d6f6bf917e6b3d7b : 20210126-1501 
    s : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/OpticksProfileLabels.npy :              (0, 64) : 4051d15b97452eb1de610547e562fe21 : 20210126-1501 
    epsilon:optixrap blyth$  np.py /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/*.npy
    a :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/dg.npy :         (2000, 1, 4) : 600d5e2b539f1aff0534bd80df0bfe78 : 20210126-1501 
    b :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ox.npy :         (2000, 4, 4) : 49576da794c90190adabe710d18cb42a : 20210126-1501 
    c :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ph.npy :         (2000, 1, 2) : 0581e94d0a5297fe54aa03b9d90c3f71 : 20210126-1501 
    d :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ps.npy :         (2000, 1, 4) : fe06a58759f74b6eec2e4ee64552be4d : 20210126-1501 
    e :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/rs.npy :     (2000, 10, 1, 4) : 1404aa74a729f9b87463f1d5e2595428 : 20210126-1501 
    f :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/rx.npy :     (2000, 10, 2, 4) : e541c4e0be35bc1d9c5e3d4175a49eda : 20210126-1501 
    g :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/wy.npy :         (2000, 2, 4) : 5386979bff5c09f1b9dfdbf9fc244a5a : 20210126-1501 
    h :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/ht.npy :           (17, 4, 4) : cdf38607f9e1086a44e306dc49472197 : 20210126-1501 
    i :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/hy.npy :           (17, 2, 4) : 5c4fcc5cd1b5e2e3cfa3d58bc498fa41 : 20210126-1501 
    j :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/fdom.npy :            (3, 1, 4) : 16b5b59871cef2abbfc9ba3499123d2d : 20210126-1501 
    k : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileAcc.npy :               (1, 4) : f949b607e29aa73b13f834396406217c : 20210126-1501 
    l : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileAccLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1501 
    m : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileLis.npy :                 (1,) : c72246e7e3772306d6e202419e22f6b0 : 20210126-1501 
    n : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileLisLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20210126-1501 
    o :      /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/gs.npy :            (1, 6, 4) : 3dd428490778dd3298816b8802d2d630 : 20210126-1501 
    p :    /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20210126-1501 
    q : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfile.npy :               (0, 4) : d1d127c8c0c63b61d6f6bf917e6b3d7b : 20210126-1501 
    r : /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/OpticksProfileLabels.npy :              (0, 64) : 4051d15b97452eb1de610547e562fe21 : 20210126-1501 
    epsilon:optixrap blyth$ 


