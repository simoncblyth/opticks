masked-running-revived
========================

Context
----------

:doc:`tboolean-proxy-scan-LV10-history-misaligned-big-bouncer`


Revive masked running 
--------------------------

* :doc:`AB_SC_Position_Time_mismatch` looks relevant for techniques


::

    ts 10 --mask 5207



    2019-06-22 21:50:29.845 INFO  [408106] [CRandomEngine::preTrack@462]  [ --align --mask ]  m_ctx._record_id:  0 mask_index: 5207 ( m_okevt_seqhis: 8cb6b6cd TO BT SC BR SC BR BT SA                          ) 
    2019-06-22 21:50:29.845 INFO  [408106] [CRandomEngine::preTrack@471] [ cmd "ucf.py 5207"
    Traceback (most recent call last):
      File "/home/blyth/opticks/ana/ucf.py", line 215, in <module>
        ucf = UCF( pindex )
      File "/home/blyth/opticks/ana/ucf.py", line 155, in __init__
        self.parse(path) 
      File "/home/blyth/opticks/ana/ucf.py", line 165, in parse
        self.lines = map(lambda line:line.rstrip(),file(path).readlines())
    IOError: [Errno 2] No such file or directory: '/tmp/blyth/location/ox_5207.log'
    2019-06-22 21:50:30.148 INFO  [408106] [SSys::run@72] ucf.py 5207 rc_raw : 256 rc : 1
    2019-06-22 21:50:30.148 ERROR [408106] [SSys::run@79] FAILED with  cmd ucf.py 5207 RC 1
    OKG4Test: /home/blyth/opticks/cfg4/CRandomEngine.cc:473: virtual void CRandomEngine::preTrack(): Assertion `rc == 0' failed.
    /home/blyth/opticks/bin/o.sh: line 183: 408106 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okg4 --a


::

   ts 10 --mask 5207 --pindex 0 --pindexlog   

          # writing empty /tmp/blyth/location/ox_5207.log


::

    2019-06-22 22:17:15.654 INFO  [453862] [CG4Ctx::setEvent@189]  gen 4096 SourceType TORCH
    2019-06-22 22:17:15.655 INFO  [453862] [CRandomEngine::preTrack@462]  [ --align --mask ]  m_ctx._record_id:  0 mask_index: 5207 ( m_okevt_seqhis: 8cb6b6cd TO BT SC BR SC BR BT SA                          ) 
    2019-06-22 22:17:15.655 INFO  [453862] [CRandomEngine::preTrack@471] [ cmd "ucf.py 5207"
    Traceback (most recent call last):
      File "/home/blyth/opticks/ana/ucf.py", line 215, in <module>
        ucf = UCF( pindex )
      File "/home/blyth/opticks/ana/ucf.py", line 155, in __init__
        self.parse(path) 
      File "/home/blyth/opticks/ana/ucf.py", line 182, in parse
        self[-1].tail = curr[:]
    IndexError: list index out of range
    2019-06-22 22:17:15.951 INFO  [453862] [SSys::run@72] ucf.py 5207 rc_raw : 256 rc : 1
    2019-06-22 22:17:15.952 ERROR [453862] [SSys::run@79] FAILED with  cmd ucf.py 5207 RC 1
    OKG4Test: /home/blyth/opticks/cfg4/CRandomEngine.cc:473: virtual void CRandomEngine::preTrack(): Assertion `rc == 0' failed.



Also need WITH_ALIGN_DEV_DEBUG switch::

    2019-06-22 22:20:32.630 INFO  [9692] [CG4Ctx::setEvent@189]  gen 4096 SourceType TORCH
    2019-06-22 22:20:32.630 INFO  [9692] [CRandomEngine::preTrack@462]  [ --align --mask ]  m_ctx._record_id:  0 mask_index: 5207 ( m_okevt_seqhis: 8cb6b6cd TO BT SC BR SC BR BT SA                          ) 
    2019-06-22 22:20:32.630 INFO  [9692] [CRandomEngine::preTrack@471] [ cmd "ucf.py 5207"
        5207 : /tmp/blyth/location/ox_5207.log  
     [  0|  0]                                         OpBoundary :      : 0.593884408 : 0.593884408 : 2 
     [  1|  1]                                         OpRayleigh :      : 0.554541230 : 0.554541230 : 1 

     ... elide as can also run from commandline, see below ...

     [ 40| 40]                   OpBoundary_DiDiReflectOrTransmit :      : 0.367534995 : 0.367534995 : 1 
     [ 41| 41]                            OpBoundary_DoAbsorption :      : 0.306532949 : 0.306532949 : 1 
    2019-06-22 22:20:32.922 INFO  [9692] [SSys::run@72] ucf.py 5207 rc_raw : 0 rc : 0
    2019-06-22 22:20:32.922 INFO  [9692] [CRandomEngine::preTrack@474] ] cmd "ucf.py 5207"
    2019-06-22 22:20:32.923 INFO  [9692] [CSensitiveDetector::EndOfEvent@110]  HCE 0xbfa60c0 hitCollectionA->entries() 0 hitCollectionB->entries() 0
    2019-06-22 22:20:32.923 INFO  [9692] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2019-06-22 22:20:32.924 INFO  [9692] [CG4::propagate@337]  calling BeamOn numG4Evt 1 DONE 



::

    [blyth@localhost optixrap]$ ucf.py 5207
        5207 : /tmp/blyth/location/ox_5207.log  
     [  0|  0]                                         OpBoundary :      : 0.593884408 : 0.593884408 : 2 
     [  1|  1]                                         OpRayleigh :      : 0.554541230 : 0.554541230 : 1 
     [  2|  2]                                       OpAbsorption :      : 0.848477900 : 0.848477900 : 1 
     [  3|  3]                          OpBoundary_DiDiTransCoeff :      : 0.427505881 : 0.427505881 : 1 
     [  4|  4]                                         OpBoundary :      : 0.487669140 : 0.487669140 : 2 
     [  5|  5]                                         OpRayleigh :      : 0.996436834 : 0.996436834 : 1 
     [  6|  6]                                       OpAbsorption :      : 0.275350600 : 0.275350600 : 1 
     [  7|  7]                                         OpRayleigh :      : 0.340107858 : 0.340107858 : 3 
     [  8|  8]                                         OpRayleigh :      : 0.265102714 : 0.265102714 : 1 
     [  9|  9]                                         OpRayleigh :      : 0.513796866 : 0.513796866 : 1 
     [ 10| 10]                                         OpRayleigh :      : 0.115229465 : 0.115229465 : 1 
     [ 11| 11]                                         OpRayleigh :      : 0.157869205 : 0.157869205 : 1 
     [ 12| 12]                                         OpBoundary :      : 0.428703308 : 0.428703308 : 6 
     [ 13| 13]                                         OpRayleigh :      : 0.458940476 : 0.458940476 : 1 
     [ 14| 14]                                       OpAbsorption :      : 0.412707955 : 0.412707955 : 1 
     [ 15| 15]                          OpBoundary_DiDiTransCoeff :      : 0.689785659 : 0.689785659 : 1 
     [ 16| 16]                                         OpBoundary :      : 0.776829183 : 0.776829183 : 2 
     [ 17| 17]                                         OpRayleigh :      : 0.987821996 : 0.987821996 : 1 
     [ 18| 18]                                       OpAbsorption :      : 0.312370330 : 0.312370330 : 1 
     [ 19| 19]                                         OpRayleigh :      : 0.397718579 : 0.397718579 : 3 
     [ 20| 20]                                         OpRayleigh :      : 0.660310566 : 0.660310566 : 1 
     [ 21| 21]                                         OpRayleigh :      : 0.571472466 : 0.571472466 : 1 
     [ 22| 22]                                         OpRayleigh :      : 0.390088499 : 0.390088499 : 1 
     [ 23| 23]                                         OpRayleigh :      : 0.937021375 : 0.937021375 : 1 
     [ 24| 24]                                         OpRayleigh :      : 0.987214506 : 0.987214506 : 5 
     [ 25| 25]                                         OpRayleigh :      : 0.550899804 : 0.550899804 : 1 
     [ 26| 26]                                         OpRayleigh :      : 0.143670171 : 0.143670171 : 1 
     [ 27| 27]                                         OpRayleigh :      : 0.379043430 : 0.379043430 : 1 
     [ 28| 28]                                         OpRayleigh :      : 0.608411908 : 0.608411908 : 1 
     [ 29| 29]                                         OpBoundary :      : 0.518446624 : 0.518446624 : 6 
     [ 30| 30]                                         OpRayleigh :      : 0.444000155 : 0.444000155 : 1 
     [ 31| 31]                                       OpAbsorption :      : 0.587975919 : 0.587975919 : 1 
     [ 32| 32]                          OpBoundary_DiDiTransCoeff :      : 0.747779906 : 0.747779906 : 1 
     [ 33| 33]                                         OpBoundary :      : 0.266604096 : 0.266604096 : 2 
     [ 34| 34]                                         OpRayleigh :      : 0.442346156 : 0.442346156 : 1 
     [ 35| 35]                                       OpAbsorption :      : 0.069202743 : 0.069202743 : 1 
     [ 36| 36]                          OpBoundary_DiDiTransCoeff :      : 0.261564046 : 0.261564046 : 1 
     [ 37| 37]                                         OpBoundary :      : 0.891545713 : 0.891545713 : 2 
     [ 38| 38]                                         OpRayleigh :      : 0.238244846 : 0.238244846 : 1 
     [ 39| 39]                                       OpAbsorption :      : 0.148877993 : 0.148877993 : 1 
     [ 40| 40]                   OpBoundary_DiDiReflectOrTransmit :      : 0.367534995 : 0.367534995 : 1 
     [ 41| 41]                            OpBoundary_DoAbsorption :      : 0.306532949 : 0.306532949 : 1 
    [blyth@localhost optixrap]$ 


::

    ts 10 --mask 5207 --pindex 0 --pindexlog

         ## also gives a propagation viz of the single photon
         ## hmm, could load up both OK and G4 photons and viz them together ?



Observations
-------------


* OptiX rtPrintf bug with bounce always being zero ? So added the #0,1, 

* TODO: time dumping in the pindexlog 



::

    In [27]: a.rposti(5207)
    Out[27]: 
    A()sliced
    A([[  5719.7364,  -3812.4252, -71998.8026,      0.    ],      TO
       [  5719.7364,  -3812.4252,  -2500.5993,    231.8218],      BT
       [  5719.7364,  -3812.4252,   1070.1159,    253.4658],      SC 
       [ -4111.2665,  -4667.1994,  -2500.5993,    317.0575],      BR
       [-15590.2919,  -5664.8023,   1667.7987,    391.3064],      SC   <--- Opticks scatters
       [-18840.1921,  -6376.748 ,   2500.5993,    412.0715],      BR
       [-23999.6009,  -7510.5874,   1177.7867,    445.0759],      BT 
       [-72001.    , -27056.1331, -21655.0143,    633.9832]])     SA

::

    [blyth@localhost issues]$ cat /tmp/blyth/location/ox_5207.log 

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0   #0
    propagate_to_boundary  u_OpBoundary:0.593884408 speed:299.79245 
    propagate_to_boundary  u_OpRayleigh:0.55454123   scattering_length(s.material1.z):1000000 scattering_distance:589614.125 
    propagate_to_boundary  u_OpAbsorption:0.8484779   absorption_length(s.material1.y):1e+09 absorption_distance:164311248 
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.427505881  reflect:0   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos ( 5720.4297 -3812.4258 -2500.0000)   



    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0   #1 
    propagate_to_boundary  u_OpBoundary:0.48766914 speed:165.028061 
    propagate_to_boundary  u_OpRayleigh:0.996436834   scattering_length(s.material1.z):1000000 scattering_distance:3569.52881 
    propagate_to_boundary  u_OpAbsorption:0.2753506   absorption_length(s.material1.y):1000000 absorption_distance:1289710.12 
    rayleigh_scatter_align p.direction (-0 -0 1) 
    rayleigh_scatter_align p.polarization (0 -1 0) 
    rayleigh_scatter_align.do u_OpRayleigh:0.340107858 
    rayleigh_scatter_align.do u_OpRayleigh:0.265102714 
    rayleigh_scatter_align.do u_OpRayleigh:0.513796866 
    rayleigh_scatter_align.do u_OpRayleigh:0.115229465 
    rayleigh_scatter_align.do u_OpRayleigh:0.157869205 
    rayleigh_scatter_align.do constant        (-0.0814180896) 
    rayleigh_scatter_align.do newDirection    (-0.936855316 -0.0814180896 -0.340107858) 
    rayleigh_scatter_align.do newPolarization (-0.07653106 0.996680081 -0.0277831722) 
    rayleigh_scatter_align.do doCosTheta -0.996680081 doCosTheta2 0.993371189   looping 0   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0    #2
    propagate_to_boundary  u_OpBoundary:0.428703308 speed:165.028061 
    propagate_to_boundary  u_OpRayleigh:0.458940476   scattering_length(s.material1.z):1000000 scattering_distance:778834.75 
    propagate_to_boundary  u_OpAbsorption:0.412707955   absorption_length(s.material1.y):1000000 absorption_distance:885015.062 
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.689785659  reflect:1   TransCoeff:   0.00000  c2c2:   -1.4361 tir:1  pos (-4112.1333 -4666.9316 -2499.9998)   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0    #3
    propagate_to_boundary  u_OpBoundary:0.776829183 speed:165.028061 
    propagate_to_boundary  u_OpRayleigh:0.987821996   scattering_length(s.material1.z):1000000 scattering_distance:12252.7637 
    propagate_to_boundary  u_OpAbsorption:0.31237033   absorption_length(s.material1.y):1000000 absorption_distance:1163565.88 
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ scattering wins easily ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
    rayleigh_scatter_align p.direction (-0.936855316 -0.0814180896 0.340107858) 
    rayleigh_scatter_align p.polarization (0.07653106 -0.996680081 -0.0277831722) 
    rayleigh_scatter_align.do u_OpRayleigh:0.397718579 
    rayleigh_scatter_align.do u_OpRayleigh:0.660310566 
    rayleigh_scatter_align.do u_OpRayleigh:0.571472466 
    rayleigh_scatter_align.do u_OpRayleigh:0.390088499 
    rayleigh_scatter_align.do u_OpRayleigh:0.937021375 
    rayleigh_scatter_align.do constant        (0.422564924) 
    rayleigh_scatter_align.do newDirection    (-0.12703526 0.388780504 0.912530422) 
    rayleigh_scatter_align.do newPolarization (-0.0252119545 0.918421149 -0.394800037) 
    rayleigh_scatter_align.do doCosTheta -0.906332791 doCosTheta2 0.821439147   looping 1   
    rayleigh_scatter_align.do u_OpRayleigh:0.987214506 
    rayleigh_scatter_align.do u_OpRayleigh:0.550899804 
    rayleigh_scatter_align.do u_OpRayleigh:0.143670171 
    rayleigh_scatter_align.do u_OpRayleigh:0.37904343 
    rayleigh_scatter_align.do u_OpRayleigh:0.608411908 
    rayleigh_scatter_align.do constant        (-0.127990544) 
    rayleigh_scatter_align.do newDirection    (-0.947501421 -0.207942754 0.242901236) 
    rayleigh_scatter_align.do newPolarization (-0.19944261 0.978109896 0.0593604483) 
    rayleigh_scatter_align.do doCosTheta -0.991775393 doCosTheta2 0.983618438   looping 0   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0  #4
    propagate_to_boundary  u_OpBoundary:0.518446624 speed:165.028061 
    propagate_to_boundary  u_OpRayleigh:0.444000155   scattering_length(s.material1.z):1000000 scattering_distance:811930.375 
    propagate_to_boundary  u_OpAbsorption:0.587975919   absorption_length(s.material1.y):1000000 absorption_distance:531069.312 
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.747779906  reflect:1   TransCoeff:   0.00000  c2c2:   -1.5922 tir:1  pos (-18839.5195 -6377.4185  2500.0000)   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 #5
    propagate_to_boundary  u_OpBoundary:0.266604096 speed:165.028061 
    propagate_to_boundary  u_OpRayleigh:0.442346156   scattering_length(s.material1.z):1000000 scattering_distance:815662.562 
    propagate_to_boundary  u_OpAbsorption:0.0692027435   absorption_length(s.material1.y):1000000 absorption_distance:2670714.75 
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.261564046  reflect:0   TransCoeff:   0.93036  c2c2:    0.7183 tir:0  pos (-24000.0000 -7509.9600  1177.0604)   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 #6
    propagate_to_boundary  u_OpBoundary:0.891545713 speed:299.79245 
    propagate_to_boundary  u_OpRayleigh:0.238244846   scattering_length(s.material1.z):1000000 scattering_distance:1434456.38 
    propagate_to_boundary  u_OpAbsorption:0.148877993   absorption_length(s.material1.y):1e+09 absorption_distance:1.9046281e+09 
    propagate_at_surface   u_OpBoundary_DiDiReflectOrTransmit:        0.367534995 
    propagate_at_surface   u_OpBoundary_DoAbsorption:   0.306532949 

     WITH_ALIGN_DEV_DEBUG psave (-72000 -27056.1016 -21655.0957 633.986206) ( 1, 0, 67305985, 7328 ) 





::

    bouncelog.py 5207
       ## just spaces out the bounces, like i just did manually 


g4lldb
-----------

* ana/g4lldb.py   `v g4lldb.py`




Hmm getting the equivalent from Geant4 is what I used g4lldb.py for 


