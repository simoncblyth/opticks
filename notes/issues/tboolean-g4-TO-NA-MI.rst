tboolean-g4-TO-NA-MI
================================

::

   ts truncate
   ta truncate
   tv truncate

   ts cone   # similar "TO NA MI" problem



Status 
----------

* "TO NA MI" issue with truncate FIXED,  few photons with deviant SC position is a different issue 



ISSUES
-----------

1. OpticksEventAna::checkPointExcursions found some
2. All G4 photons are  "TO NA MI"  NA:NAN-ABORT MI:MISS


ts truncate::

    ...
    2019-06-24 16:43:57.541 INFO  [31414] [OpticksEvent::makeReport@1688] tagdir /tmp/blyth/opticks/tboolean-truncate/evt/tboolean-truncate/torch/1
    2019-06-24 16:43:57.545 INFO  [31414] [GGeoTest::anaEvent@804]  dbgnode -1 numTrees 1 evt 0xa66cb60
    2019-06-24 16:43:57.545 INFO  [31414] [OpticksEvent::getTestConfig@706]  gtc autoseqmap=TO:0,SR:1,SA:0_name=tboolean-truncate_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-truncate_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75_autocontainer=Rock//perfectAbsorbSurface/Vacuum
    2019-06-24 16:43:57.546 INFO  [31414] [OpticksEventInstrument::CreateRecordsNPY@36] OpticksEventInstrument::CreateRecordsNPY  shape 100000,10,2,4
    2019-06-24 16:43:57.546 INFO  [31414] [OpticksEventAna::initOverride@67]  autoseqmap TO:0,SR:1,SA:0
    2019-06-24 16:43:57.549 INFO  [31414] [OpticksEventAna::checkPointExcursions@108]  seqmap TO:0,SR:1,SA:0 seqmap_his              8ad seqmap_val              121
     p  0 abbrev TO val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (     -0.100    -0.100    -0.100      -0.100) df 0.1000000015 expected
     p  1 abbrev SR val1  2 tree 1 count 3276275712 dist (   -200.000   200.000  -150.000    -200.000) xdist (      0.000     0.000     0.000       0.000) df 200.0000000000 EXCURSION
     p  2 abbrev SA val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (      0.000     0.000     0.000       0.000) df 0.0000000000 expected
    2019-06-24 16:43:57.549 FATAL [31414] [OpticksEventAna::checkPointExcursions@157]  num_excursions 1
    2019-06-24 16:43:57.549 FATAL [31414] [Opticks::dumpRC@202]  rc 202 rcmsg : OpticksEventAna::checkPointExcursions found some
    2019-06-24 16:43:57.549 INFO  [31414] [OpticksEventAna::dump@232] GGeoTest::anaEvent OpticksEventAna pho 100000,4,4 seq 100000,1,2
    2019-06-24 16:43:57.549 INFO  [31414] [GGeoTest::anaEvent@804]  dbgnode -1 numTrees 1 evt 0xac2c230
    2019-06-24 16:43:57.549 INFO  [31414] [OpticksEvent::getTestConfig@706]  gtc autoseqmap=TO:0,SR:1,SA:0_name=tboolean-truncate_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-truncate_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75_autocontainer=Rock//perfectAbsorbSurface/Vacuum
    2019-06-24 16:43:57.550 INFO  [31414] [OpticksEventInstrument::CreateRecordsNPY@36] OpticksEventInstrument::CreateRecordsNPY  shape 100000,10,2,4
    2019-06-24 16:43:57.550 INFO  [31414] [OpticksEventAna::initOverride@67]  autoseqmap TO:0,SR:1,SA:0
    2019-06-24 16:43:57.553 INFO  [31414] [OpticksEventAna::checkPointExcursions@108]  seqmap TO:0,SR:1,SA:0 seqmap_his              8ad seqmap_val              121
     p  0 abbrev TO val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (     -0.100    -0.100    -0.100      -0.100) df 0.1000000015 expected
     p  1 abbrev SR val1  2 tree 1 count 3276275712 dist (   -200.000   200.000  -150.000    -200.000) xdist (      0.000     0.000     0.000       0.000) df 200.0000000000 EXCURSION
     p  2 abbrev SA val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (      0.000     0.000     0.000       0.000) df 0.0000000000 expected
    2019-06-24 16:43:57.553 FATAL [31414] [OpticksEventAna::checkPointExcursions@157]  num_excursions 1
    2019-06-24 16:43:57.553 FATAL [31414] [Opticks::dumpRC@202]  rc 202 rcmsg : OpticksEventAna::checkPointExcursions found some
    2019-06-24 16:43:57.553 INFO  [31414] [OpticksEventAna::dump@232] GGeoTest::anaEvent OpticksEventAna pho 100000,4,4 seq 100000,1,2
    2019-06-24 16:43:57.553 INFO  [31414] [OpticksAna::run@70]  anakey tboolean enabled Y

    args: /home/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --det tboolean-truncate --pfx tboolean-truncate --src torch
    [2019-06-24 16:43:58,404] p31868 {tboolean.py:63} INFO     - pfx tboolean-truncate tag 1 src torch det tboolean-truncate c2max [1.5, 2.0, 2.5] ipython False 
    [2019-06-24 16:43:58,412] p31868 {evt.py    :446} WARNING  -  x : -200.000 200.000 : tot 100000 over 10 0.000  under 10 0.000 : mi   -200.000 mx    200.000  
    [2019-06-24 16:43:58,413] p31868 {evt.py    :446} WARNING  -  y : -200.000 200.000 : tot 100000 over 1 0.000  under 5 0.000 : mi   -200.000 mx    200.000  
    [2019-06-24 16:43:58,415] p31868 {evt.py    :446} WARNING  -  z : -200.000 200.000 : tot 100000 over 11 0.000  under 14 0.000 : mi   -200.000 mx    200.000  
    [2019-06-24 16:43:58,416] p31868 {evt.py    :446} WARNING  -  t :   0.000   4.000 : tot 100000 over 100000 1.000  under 0 0.000 : mi      6.181 mx     12.239  
    [2019-06-24 16:43:58,458] p31868 {evt.py    :596} WARNING  - init_records tboolean-truncate/tboolean-truncate/torch/  1 :  finds too few (ph)seqmat uniques : 1 : EMPTY HISTORY
    ab.cfm
    nph:  100000 A:    0.0117 B:    9.3984 B/A:     802.0 INTEROP_MODE ALIGN non-reflectcheat 
    ab.a.metadata:/tmp/blyth/opticks/tboolean-truncate/evt/tboolean-truncate/torch/1 ox:c38f1bd703797b74e0396028b7912809 rx:9c8e93970c6237f9ca465d276eb38933 np: 100000 pr:    0.0117 INTEROP_MODE
    ab.b.metadata:/tmp/blyth/opticks/tboolean-truncate/evt/tboolean-truncate/torch/-1 ox:c1612472d50a26b8d5f2b0bf2d6d526c rx:7d8577bba5bc33b2311aa65a800cc21e np: 100000 pr:    9.3984 INTEROP_MODE
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_ALIGN_DEV_DEBUG WITH_LOGDOUBLE 
    {u'containerscale': 3.0, u'ctrl': 0, u'verbosity': 0, u'poly': u'IM', u'jsonLoadPath': u'/tmp/blyth/opticks/tboolean-truncate/0/meta.json', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1', u'resolution': 20, u'emit': -1}
    .
    ab.mal
    aligned        0/ 100000 : 0.0000 :  
    maligned  100000/ 100000 : 1.0000 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    slice(0, 25, None)
          0      0 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          1      1 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          2      2 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          3      3 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          4      4 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          5      5 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          6      6 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          7      7 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          8      8 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          9      9 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
         10     10 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
         11     11 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
         12     12 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 



tboolean-truncat basic comparisons
------------------------------------

Just a box with one face emissive with photons bouncing up and down::

    tboolean-truncate--
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.main import opticks_main
    from opticks.analytic.polyconfig import PolyConfig
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgname="tboolean-truncate")

    emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1" 

    CSG.kwa = dict(poly="IM",resolution=20, verbosity=0,ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

    box = CSG("box", param=[0,0,0,200], emit=-1,  boundary="Rock//perfectSpecularSurface/Vacuum" )

    CSG.Serialize([box], args )



::

   tv truncate 
   tv4 truncate 



posdelta=0.1 looks to be an offset to make sure the emitted photons dont immediately intersect 
with the surface from whence they came, and it looks to not be working for Geant4.  

Increasing it to 0.5 makes no difference.



G4::

    In [1]: b.rpostn(3)
    Out[1]: 
    A()sliced
    A([[[  50.1846, -153.9781, -199.9023,    0.2   ],
        [  50.1846, -153.9781, -199.9023,    0.2   ],
        [  50.1846, -153.9781,  200.    ,    1.534 ]],

       [[-118.6193,   -1.2574, -199.9023,    0.2   ],
        [-118.6193,   -1.2574, -199.9023,    0.2   ],
        [-118.6193,   -1.2574,  200.    ,    1.534 ]],

       [[-180.285 ,   43.6293, -199.9023,    0.2   ],
        [-180.285 ,   43.6293, -199.9023,    0.2   ],
        [-180.285 ,   43.6293,  200.    ,    1.534 ]],

       ...,

       [[-141.0382, -123.3192, -199.9023,    0.2   ],
        [-141.0382, -123.3192, -199.9023,    0.2   ],
        [-141.0382, -123.3192,  200.    ,    1.534 ]],

       [[-144.6821,   35.9203, -199.9023,    0.2   ],
        [-144.6821,   35.9203, -199.9023,    0.2   ],
        [-144.6821,   35.9203,  200.    ,    1.534 ]],

       [[-149.8886,  -36.6955, -199.9023,    0.2   ],
        [-149.8886,  -36.6955, -199.9023,    0.2   ],
        [-149.8886,  -36.6955,  200.    ,    1.534 ]]])


OK::

    In [11]: a.rpost_(slice(0,5))
    Out[11]: 
    A()sliced
    A([[[  50.1846, -153.9781, -199.9023,    0.2   ],    
        [  50.1846, -153.9781,  200.    ,    1.534 ],
        [  50.1846, -153.9781, -200.    ,    2.8681],
        [  50.1846, -153.9781,  200.    ,   -4.0001],
        [  50.1846, -153.9781, -200.    ,   -4.0001]],

       [[-118.6193,   -1.2574, -199.9023,    0.2   ],
        [-118.6193,   -1.2574,  200.    ,    1.534 ],
        [-118.6193,   -1.2574, -200.    ,    2.8681],
        [-118.6193,   -1.2574,  200.    ,   -4.0001],
        [-118.6193,   -1.2574, -200.    ,   -4.0001]],

       [[-180.285 ,   43.6293, -199.9023,    0.2   ],
        [-180.285 ,   43.6293,  200.    ,    1.534 ],
        [-180.285 ,   43.6293, -200.    ,    2.8681],
        [-180.285 ,   43.6293,  200.    ,   -4.0001],
        [-180.285 ,   43.6293, -200.    ,   -4.0001]],






    In [2]: b.rpostn(3).shape
    Out[2]: (99962, 3, 4)

    In [3]: b.rpostn(4).shape
    Out[3]: (38, 4, 4)

    In [4]: b.rpostn(4)
    Out[4]: 
    A()sliced
    A([[[ -19.4952,   77.0287, -199.9023,    0.2   ],
        [ -19.4952,   77.0287, -199.9023,    0.2   ],
        [ -19.4952,   77.0287,  -23.2734,    0.7892],
        [ 200.    ,  195.0011,  -17.9632,    1.6205]],

       [[-186.1019,  177.7398, -199.9023,    0.2   ],
        [-186.1019,  177.7398, -199.9023,    0.2   ],
        [-186.1019,  177.7398,   33.9549,    0.98  ],
        [ 177.5994,  154.7227,  200.    ,    2.3159]],

       [[ 183.6543,  144.615 , -199.9023,    0.2   ],
        [ 183.6543,  144.615 , -199.9023,    0.2   ],
        [ 183.6543,  144.615 ,  127.2011,    1.2911],
        [ 112.0579,  200.    ,  118.3203,    1.5945]],





* getting MI rather than a bounce as do not have a universe wrapper perhaps ?



::

    311 unsigned int OpStatus::OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst, CStage::CStage_t stage)
    312 #endif
    313 {
    314     G4StepStatus status = point->GetStepStatus()  ;
    315     // TODO: cache the relevant process objects, so can just compare pointers ?
    316     const G4VProcess* process = point->GetProcessDefinedStep() ;
    317     const G4String& processName = process ? process->GetProcessName() : "NoProc" ;
    318 
    319     bool transportation = strcmp(processName,"Transportation") == 0 ;
    320     bool scatter = strcmp(processName, "OpRayleigh") == 0 ;
    321     bool absorption = strcmp(processName, "OpAbsorption") == 0 ;
    322 
    323     unsigned flag(0);
    324 
    325     // hmm stage and REJOINing look kinda odd here, do elsewhere ?
    326     // moving it first, breaks seqhis matching for multi-RE lines 
    327 
    328     if(absorption && status == fPostStepDoItProc )
    329     {
    330         flag = BULK_ABSORB ;
    331     }
    332     else if(scatter && status == fPostStepDoItProc )
    333     {
    334         flag = BULK_SCATTER ;
    335     }
    336     else if( stage == CStage::REJOIN )
    337     {
    338         flag = BULK_REEMIT ; 
    339     }
    340     else if(transportation && status == fGeomBoundary )
    341     {
    342         flag = OpStatus::OpBoundaryFlag(bst) ; // BOUNDARY_TRANSMIT/BOUNDARY_REFLECT/NAN_ABORT/SURFACE_ABSORB/SURFACE_DETECT/SURFACE_DREFLECT/SURFACE_SREFLECT
    343     }
    344     else if(transportation && status == fWorldBoundary )
    345     {
    346         //flag = SURFACE_ABSORB ;   // former kludge for fWorldBoundary - no surface handling yet 
    347         flag = MISS ;
    348     }
    349     else
    350     {
    351         LOG(warning) << " OpPointFlag ZERO  "
    352                      << " proceesDefinedStep? " << processName
    353                      << " stage " << CStage::Label(stage)
    354                      << " status " << OpStepString(status)
    355                      ;
    356         assert(0);
    357     }
    358     return flag ;
    359 }




NAN_ABORT means are getting StepTooSmall::


    201 #ifdef USE_CUSTOM_BOUNDARY
    202 unsigned int OpStatus::OpBoundaryFlag(const Ds::DsG4OpBoundaryProcessStatus status)
    203 {
    204     unsigned flag = 0 ;
    205     switch(status)
    206     {
    207         case Ds::FresnelRefraction:
    208         case Ds::SameMaterial:
    209                                flag=BOUNDARY_TRANSMIT;
    210                                break;
    211         case Ds::TotalInternalReflection:
    212         case Ds::FresnelReflection:
    213                                flag=BOUNDARY_REFLECT;
    214                                break;
    215         case Ds::StepTooSmall:
    216                                flag=NAN_ABORT;
    217                                break;
    218         case Ds::Absorption:




Reference to 

* :doc:`cfg4-bouncemax-not-working`

* :doc:`geant4_opticks_integration/tconcentric_pflags_mismatch_from_truncation_handling`





Curious adding a small box and making the outer box a container
makes things behave much more reasonably in "--noalign"::

    ts truncate --noalign


::

    tboolean-truncate--(){ cat << EOP 
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.main import opticks_main
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgname="${FUNCNAME/--}")

    emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.5,sheetmask:0x1" 

    CSG.kwa = dict(poly="IM",resolution=20, verbosity=0,ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

    smallbox = CSG("box", param=[0,0,0,10], emit=0,  boundary="Vacuum///Water"  )

    box = CSG("box", param=[0,0,0,200], emit=-1,  boundary="Rock//perfectSpecularSurface/Vacuum", container=1  )

    CSG.Serialize([box, smallbox], args )
    EOP
    }




::

    (gdb) bt
    #0  0x00007fffe200d207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe200e8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2006026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20060d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdf4711 in CRandomEngine::_peek (this=0x6a99cb0, offset=-2) at /home/blyth/opticks/cfg4/CRandomEngine.cc:289
    #5  0x00007fffefdf44f0 in CRandomEngine::flat (this=0x6a99cb0) at /home/blyth/opticks/cfg4/CRandomEngine.cc:255
    #6  0x00007fffeaf9606e in G4VProcess::ResetNumberOfInteractionLengthLeft (this=0x6e29d40) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/processes/management/src/G4VProcess.cc:98
    #7  0x00007fffeaf95904 in G4VDiscreteProcess::PostStepGetPhysicalInteractionLength (this=0x6e29d40, track=..., previousStepSize=0, condition=0x6c150c8)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/processes/management/src/G4VDiscreteProcess.cc:79
    #8  0x00007fffec1289b2 in G4VProcess::PostStepGPIL (this=0x6e29d40, track=..., previousStepSize=0, condition=0x6c150c8) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/processes/management/include/G4VProcess.hh:506
    #9  0x00007fffec127161 in G4SteppingManager::DefinePhysicalStepLength (this=0x6c14f40) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/tracking/src/G4SteppingManager2.cc:175
    #10 0x00007fffec124410 in G4SteppingManager::Stepping (this=0x6c14f40) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/tracking/src/G4SteppingManager.cc:180
    #11 0x00007fffec130236 in G4TrackingManager::ProcessOneTrack (this=0x6c14f00, apValueG4Track=0xdbbb190) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/tracking/src/G4TrackingManager.cc:126
    #12 0x00007fffec3a8d46 in G4EventManager::DoProcessing (this=0x6c14e70, anEvent=0x8fcb6f0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:185
    #13 0x00007fffec3a9572 in G4EventManager::ProcessOneEvent (this=0x6c14e70, anEvent=0x8fcb6f0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:338
    #14 0x00007fffec6ab665 in G4RunManager::ProcessOneEvent (this=0x6a9a110, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:399
    #15 0x00007fffec6ab4d7 in G4RunManager::DoEventLoop (this=0x6a9a110, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #16 0x00007fffec6aad2d in G4RunManager::BeamOn (this=0x6a9a110, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #17 0x00007fffefde9f9c in CG4::propagate (this=0x6a99a80) at /home/blyth/opticks/cfg4/CG4.cc:335
    #18 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffcc50) at /home/blyth/opticks/okg4/OKG4Mgr.cc:177
    #19 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcc50) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #20 0x00000000004039a9 in main (argc=32, argv=0x7fffffffcf88) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) f 4
    #4  0x00007fffefdf4711 in CRandomEngine::_peek (this=0x6a99cb0, offset=-2) at /home/blyth/opticks/cfg4/CRandomEngine.cc:289
    289     assert( idx >= 0 && idx < int(m_sequence.size()) );
    (gdb) p idx
    $1 = -3
    (gdb) p m_sequence.size()
    $2 = 256
    (gdb) 



Adding WITH_KLUDGE_FLAT_ZERO_NOPEEK as apparently the value is not used, succeed to run aligned.
And agreement is good, only deviation warnings.  Container must be doing something to keep Geant4 peachy.


After fixing issues with container sizing, that were introduced with the proxy handling 
tboolean-truncate is now working without the help of the small box,


A small number of photons are deciding to scatter in different positions::


    b.rpost_dv
    maxdvmax:3600.0213  level:FATAL  RC:1       skip:
                     :                                :                   :                       :                   : 0.0550 0.0824 0.1099 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :  TO SR SR SR SR SR SR SR SR SR :   96781    96821  :       94051   3762040 :  1210     0     0 : 0.0003 0.0000 0.0000 :    0.0550    0.0000    0.0000   :              WARNING :   > dvmax[0] 0.0550  
     0001            :  TO SR SR SC SR SR SR SR SR SR :     392      342  :           2        80 :    28    28    28 : 0.3500 0.3500 0.3500 : 2373.9430    0.0000  307.1893   :                FATAL :   > dvmax[2] 0.1099  
     0002            :  TO SR SR SR SR SC SR SR SR SR :     374      350  :           3       120 :    27    27    27 : 0.2250 0.2250 0.2250 : 3600.0213    0.0000  153.7714   :                FATAL :   > dvmax[2] 0.1099  
     0005            :  TO SR SR SR SR SR SR SC SR SR :     357      364  :           1        40 :     3     3     3 : 0.0750 0.0750 0.0750 : 1264.9377    0.0000   94.8690   :                FATAL :   > dvmax[2] 0.1099  
     0006            :  TO SC SR SR SR SR SR SR SR SR :     346      343  :         337     13480 :   106     0     0 : 0.0079 0.0000 0.0000 :    0.0550    0.0000    0.0004   :              WARNING :   > dvmax[0] 0.0550  
     0009            :  TO SR SC SR SR SR SR SR SR SR :     331      353  :           2        80 :    32    32    32 : 0.4000 0.4000 0.4000 : 3600.0213    0.0000  198.7817   :                FATAL :   > dvmax[2] 0.1099  
    .






* the -ve rpost times are from insufficient time domain from all those bounces

::

    n [1]: ab.aselhis = "TO SR SR SC SR SR SR SR SR SR"

    In [2]: a.rpost().shape
    Out[2]: (2, 10, 4)

    In [3]: a.rpost()
    Out[3]: 
    A()sliced
    A([[[-1107.7411, -1751.9723, -1799.516 ,     0.    ],
        [-1107.7411, -1751.9723,  1800.0107,    12.0063],
        [-1107.7411, -1751.9723, -1800.0107,    24.0148],
        [-1107.7411, -1751.9723,  -794.3374,    27.3698],
        [-1800.0107, -1215.1954,  -628.6764,    30.3433],
        [ 1800.0107,  1576.2529,   232.552 ,   -36.0211],
        [ 1511.3956,  1800.0107,   301.5866,   -36.0211],
        [-1800.0107,  -767.6249,  1093.8353,   -36.0211],
        [ -468.6217, -1800.0107,  1412.3507,   -36.0211],
        [ 1151.7122,  -543.5923,  1800.0107,   -36.0211]],

       [[ -103.8267, -1686.2355, -1799.516 ,     0.    ],
        [ -103.8267, -1686.2355,  1800.0107,    12.0063],
        [ -103.8267, -1686.2355, -1800.0107,    24.0148],
        [ -103.8267, -1686.2355,  1079.05  ,    33.6181],
        [-1800.0107, -1283.6804,     2.4184,   -36.0211],
        [ 1039.6959,  -609.8238, -1800.0107,   -36.0211],
        [ 1800.0107,  -429.3775, -1317.4282,   -36.0211],
        [-1800.0107,   424.9804,   967.6383,   -36.0211],
        [ -488.6285,   736.1856,  1800.0107,   -36.0211],
        [ 1800.0107,  1279.2833,   347.3165,   -36.0211]]])

    In [4]: b.rpost()
    Out[4]: 
    A()sliced
    A([[[-1107.7411, -1751.9723, -1799.516 ,     0.    ],
        [-1107.7411, -1751.9723,  1800.0107,    12.0063],
        [-1107.7411, -1751.9723, -1800.0107,    24.0148],
        [-1107.7411, -1751.9723,   479.7793,    31.6196],
        [-1800.0107, -1215.1954,   645.3854,    34.5931],
        [ 1800.0107,  1576.2529,  1506.6687,   -36.0211],
        [ 1511.3956,  1800.0107,  1575.7032,   -36.0211],
        [  573.9324,  1073.1139,  1800.0107,   -36.0211],
        [-1800.0107,  -767.6249,  1232.0693,   -36.0211],
        [ -468.6217, -1800.0107,   913.5539,   -36.0211]],

       [[ -103.8267, -1686.2355, -1799.516 ,     0.    ],
        [ -103.8267, -1686.2355,  1800.0107,    12.0063],
        [ -103.8267, -1686.2355, -1800.0107,    24.0148],
        [ -103.8267, -1686.2355,   153.1842,    30.5302],
        [-1800.0107, -1283.6804,  -923.4474,   -36.0211],
        [ -418.9893,  -955.986 , -1800.0107,   -36.0211],
        [ 1800.0107,  -429.3775,  -391.5624,   -36.0211],
        [-1652.7076,   390.0234,  1800.0107,   -36.0211],
        [-1800.0107,   424.9804,  1706.5172,   -36.0211],
        [ 1800.0107,  1279.2833,  -578.5493,   -36.0211]]])

    In [5]: a.rpost() - b.rpost()
    Out[5]: 
    A()sliced
    A([[[    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    , -1274.1167,    -4.2498],
        [    0.    ,     0.    , -1274.0617,    -4.2498],
        [    0.    ,     0.    , -1274.1167,     0.    ],
        [    0.    ,     0.    , -1274.1167,     0.    ],
        [-2373.943 , -1840.7389,  -706.1754,     0.    ],
        [ 1331.389 , -1032.3857,   180.2814,     0.    ],
        [ 1620.3339,  1256.4183,   886.4567,     0.    ]],

       [[    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,   925.8658,     3.0879],
        [    0.    ,     0.    ,   925.8658,     0.    ],
        [ 1458.6852,   346.1622,     0.    ,     0.    ],
        [    0.    ,     0.    ,  -925.8658,     0.    ],
        [ -147.3031,    34.957 ,  -832.3723,     0.    ],
        [ 1311.3822,   311.2052,    93.4935,     0.    ],
        [    0.    ,     0.    ,   925.8658,     0.    ]]])

    In [6]: 

