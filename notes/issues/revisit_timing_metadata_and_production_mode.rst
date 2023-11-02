revisit_timing_metadata_and_production_mode
============================================

::

    ./G4CXTest_raindrop.sh
    CVD=0 ./G4CXTest_raindrop.sh
    CVD=1 ./G4CXTest_raindrop.sh


No CVD : before adding extra stamps
--------------------------------------

::

    In [2]: a.f.NPFold_meta
    Out[2]: 
    source:G4CXOpticks::init_SEvt
    creator:G4CXTest
    stamp:1698896834808484
    stampFmt:2023-11-02T11:47:14.808484
    uname:Linux localhost.localdomain 3.10.0-957.10.1.el7.x86_64 #1 SMP Mon Mar 18 15:06:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
    HOME:/home/blyth
    USER:blyth
    PWD:/data/blyth/junotop/opticks/g4cx/tests
    VERSION:0
    GEOM:RaindropRockAirWater
    ${GEOM}_GEOMList:RaindropRockAirWater_GEOMList
    GPUMeta:0:TITAN_V 1:TITAN_RTX
    C4Version:0.1.9
    t_BeginOfEvent:1698896873344556
    t_EndOfEvent:1698896873384928
    t_Event:40372
    t_Launch:0.0191458
    T_BeginOfRun:1698896836244512

::

    In [1]: a.ettc
    Out[1]: 
    array([['bor', '1698910756743540', '2023-11-02T07:39:16.743540', '0.0', '0'],
           ['boe', '1698910794111425', '2023-11-02T07:39:54.111425', '37.367885', '37367885'],
           ['pre', '1698910794133058', '2023-11-02T07:39:54.133058', '0.021633', '21633'],
           ['pos', '1698910794152469', '2023-11-02T07:39:54.152469', '0.019411', '19411'],
           ['eoe', '1698910794152482', '2023-11-02T07:39:54.152482', '1.3e-05', '13']], dtype='<U32')


* boe: begin-of-event
* pre: pre-launch
* pos: post-launch

Looks like upload taking same time as the launch here. 

* HMM : curious how long the full point recording and download are taking


Add some more stamps in QEvent::setGenstep
---------------------------------------------


::

    In [1]: a.ettc
    Out[1]: 
    array([['bor', '1698915722806761', '2023-11-02T09:02:02.806761', '0.0', '0'],
           ['boe', '1698915759325602', '2023-11-02T09:02:39.325602', '36.518841', '36518841'],
           ['gs0', '1698915759325677', '2023-11-02T09:02:39.325677', '7.5e-05', '75'],
           ['gs1', '1698915759356331', '2023-11-02T09:02:39.356331', '0.030654', '30654'],
           ['pre', '1698915759356332', '2023-11-02T09:02:39.356332', '1e-06', '1'],
           ['pos', '1698915759376891', '2023-11-02T09:02:39.376891', '0.020559', '20559'],
           ['eoe', '1698915759376903', '2023-11-02T09:02:39.376903', '1.2e-05', '12']], dtype='<U32')

    In [2]: a.f.NPFold_meta
    Out[2]: 
    source:G4CXOpticks::init_SEvt
    creator:G4CXTest
    stamp:1698915721502080
    stampFmt:2023-11-02T17:02:01.502080
    uname:Linux localhost.localdomain 3.10.0-957.10.1.el7.x86_64 #1 SMP Mon Mar 18 15:06:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
    CVD:1
    CUDA_VISIBLE_DEVICES:1
    HOME:/home/blyth
    USER:blyth
    PWD:/data/blyth/junotop/opticks/g4cx/tests
    VERSION:0
    GEOM:RaindropRockAirWater
    ${GEOM}_GEOMList:RaindropRockAirWater_GEOMList
    GPUMeta:1:TITAN_RTX
    C4Version:0.1.9
    T_BeginOfRun:1698915722806761
    t_BeginOfEvent:1698915759325602
    t_setGenstep0:1698915759325677
    t_setGenstep1:1698915759356331
    t_PreLaunch:1698915759356332
    t_PostLaunch:1698915759376891
    t_EndOfEvent:1698915759376903
    t_Event:51301
    t_Launch:0.0204917



    In [1]: a.f.NPFold_meta
    Out[1]: 
    source:G4CXOpticks::init_SEvt
    creator:G4CXTest
    stamp:1698916014033478
    stampFmt:2023-11-02T17:06:54.033478
    uname:Linux localhost.localdomain 3.10.0-957.10.1.el7.x86_64 #1 SMP Mon Mar 18 15:06:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
    CVD:0
    CUDA_VISIBLE_DEVICES:0
    HOME:/home/blyth
    USER:blyth
    PWD:/data/blyth/junotop/opticks/g4cx/tests
    VERSION:0
    GEOM:RaindropRockAirWater
    ${GEOM}_GEOMList:RaindropRockAirWater_GEOMList
    GPUMeta:0:TITAN_V
    C4Version:0.1.9
    T_BeginOfRun:1698916015373360
    t_BeginOfEvent:1698916052794386
    t_setGenstep0:1698916052794435
    t_setGenstep1:1698916052817666
    t_PreLaunch:1698916052817667
    t_PostLaunch:1698916052836852
    t_EndOfEvent:1698916052836866
    t_Event:42480
    t_Launch:0.0190826

    In [2]: a.ettc
    Out[2]: 
    array([['bor', '1698916015373360', '2023-11-02T09:06:55.373360', '0.0', '0'],
           ['boe', '1698916052794386', '2023-11-02T09:07:32.794386', '37.421026', '37421026'],
           ['gs0', '1698916052794435', '2023-11-02T09:07:32.794435', '4.9e-05', '49'],
           ['gs1', '1698916052817666', '2023-11-02T09:07:32.817666', '0.023231', '23231'],
           ['pre', '1698916052817667', '2023-11-02T09:07:32.817667', '1e-06', '1'],
           ['pos', '1698916052836852', '2023-11-02T09:07:32.836852', '0.019185', '19185'],
           ['eoe', '1698916052836866', '2023-11-02T09:07:32.836866', '1.4e-05', '14']], dtype='<U32')



HUH : the genstep is small ? Cannot be the upload ? Probably memset up to maxphoton ? 
-----------------------------------------------------------------------------------------

Added more stamps to pin it down. 

::

    N[blyth@localhost p001]$ du -hs genstep.npy 
    4.0K	genstep.npy


::

    In [1]: a.ettc
    Out[1]: 
    array([['bor', '1698917834971368', '2023-11-02T09:37:14.971368', '0.0', '0'],
           ['boe', '1698917872407257', '2023-11-02T09:37:52.407257', '37.435889', '37435889'],
           ['gs0', '1698917872407305', '2023-11-02T09:37:52.407305', '4.8e-05', '48'],
           ['gs1', '1698917872407362', '2023-11-02T09:37:52.407362', '5.7e-05', '57'],
           ['gs2', '1698917872407409', '2023-11-02T09:37:52.407409', '4.7e-05', '47'],
           ['gs3', '1698917872407420', '2023-11-02T09:37:52.407420', '1.1e-05', '11'],
           ['gs4', '1698917872408901', '2023-11-02T09:37:52.408901', '0.001481', '1481'],
           ['gs5', '1698917872413325', '2023-11-02T09:37:52.413325', '0.004424', '4424'],
           ['gs6', '1698917872413454', '2023-11-02T09:37:52.413454', '0.000129', '129'],
           ['gs7', '1698917872414263', '2023-11-02T09:37:52.414263', '0.000809', '809'],
           ['gs8', '1698917872425257', '2023-11-02T09:37:52.425257', '0.010994', '10994'],
           ['pre', '1698917872425258', '2023-11-02T09:37:52.425258', '1e-06', '1'],
           ['pos', '1698917872444482', '2023-11-02T09:37:52.444482', '0.019224', '19224'],
           ['eoe', '1698917872444498', '2023-11-02T09:37:52.444498', '1.6e-05', '16']], dtype='<U32')

    In [2]: a.f.NPFold_meta


* gs3->gs4 QEvent::device_alloc_genstep_and_seed 
* gs7->gs8 QEvent::setNumPhoton that allocates GPU buffers for first event 


* lots of the overhead probably for first event only 







TODO : PRODUCTION switch needs to be in the NPFold metadata
--------------------------------------------------------------

::

    epsilon:opticks blyth$ opticks-fl PRODUCTION
    ./CSGOptiX/CSGOptiX7.cu
    ./sysrap/sctx.h
    ./sysrap/sevent.h
    ./sysrap/SEvt.cc
    ./qudarap/qsim.h
    ./u4/U4Recorder.cc
    ./optickscore/Opticks.cc
    epsilon:opticks blyth$ 



