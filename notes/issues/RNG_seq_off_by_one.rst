RNG_seq_off_by_one
===================


Launch
--------

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog -DD   


    ucf.py 9041

    tboolean-;tboolean-box --okg4 --align --mask 9041 --pindex 0 --pindexlog -DD 


Where consumption is mis-aligned
---------------------------------

::

    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = FresnelReflection 
    flatExit: mrk:   crfc:    5 df:3.44848594e-11 flat:0.753801465  ufval:0.753801465 :          OpBoundary; : lufc : 29    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1
    propagate_to_boundary  u_boundary_burn:0.753801465 speed:299.79245
     [  4]                                      boundary_burn :     0.753801465 :    : 0.753801465 : 0.753801465 : 2 

    flatExit: mrk:   crfc:    6 df:4.58282523e-10 flat:0.999846756  ufval:0.999846756 :          OpRayleigh; : lufc : 29    
    propagate_to_boundary  u_scattering:0.999846756   scattering_length(s.material1.z):1000000 scattering_distance:153.25528
     [  5]                                         scattering :     0.999846756 :    : 0.999846756 : 0.999846756 : 1 

    flatExit: mrk:   crfc:    7 df:3.11492943e-10 flat:0.438019574  ufval:0.438019574 :        OpAbsorption; : lufc : 29    
    propagate_to_boundary  u_absorption:0.438019574   absorption_length(s.material1.y):10000000 absorption_distance:8254917
     [  6]                                         absorption :     0.438019574 :    : 0.438019574 : 0.438019574 : 1 

    2017-12-14 20:10:31.576 INFO  [601836] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-14 20:10:31.576 ERROR [601836] [CRandomEngine::poststep@233] CRandomEngine::poststep _noZeroSteps 1 backseq -3
    flatExit: mrk:** crfc:    8 df:0.039769888 flat:0.753801465  ufval:0.714031577 :          OpBoundary; : lufc : 29    
    rayleigh_scatter_align p.direction (0 0 -1)
    rayleigh_scatter_align p.polarization (-0 1 -0)
    rayleigh_scatter_align.do u_rsa0:0.714031577
     [  7]                                               rsa0 :     0.714031577 :    : 0.714031577 : 0.714031577 : 3 

    Process 16107 stopped
    * thread #1: tid = 0x92eec, 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x000000010c738070) + 1082 at CRandomEngine.cc:203, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x000000010c738070) + 1082 at CRandomEngine.cc:203
       200      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       201      m_current_record_flat_count++ ; 
       202      m_current_step_flat_count++ ; 
    -> 203      return m_flat ;   // (*lldb*) flatExit
       204  }
       205  
       206  
    (lldb) bt
    * thread #1: tid = 0x92eec, 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x000000010c738070) + 1082 at CRandomEngine.cc:203, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x000000010c738070) + 1082 at CRandomEngine.cc:203
        frame #1: 0x0000000105b32b17 libG4processes.dylib`G4VProcess::ResetNumberOfInteractionLengthLeft(this=0x00000001109c53d0) + 23 at G4VProcess.cc:97
        frame #2: 0x0000000105b32472 libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=<unavailable>, track=<unavailable>, previousStepSize=<unavailable>, condition=<unavailable>) + 82 at G4VDiscreteProcess.cc:79
        frame #3: 0x0000000105291d67 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength() [inlined] G4VProcess::PostStepGPIL(this=0x00000001109c53d0, track=<unavailable>, previousStepSize=<unavailable>, condition=<unavailable>) + 14 at G4VProcess.hh:503
        frame #4: 0x0000000105291d59 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000110943920) + 249 at G4SteppingManager2.cc:172
        frame #5: 0x000000010529073e libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110943920) + 366 at G4SteppingManager.cc:180
        frame #6: 0x000000010529a771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x00000001109438e0, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #7: 0x00000001051f2727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110943850, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #8: 0x0000000105174611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010c738420, i_event=0) + 49 at G4RunManager.cc:399
        frame #9: 0x00000001051744db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010c738420, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #10: 0x0000000105173913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010c738420, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #11: 0x00000001044d8dd6 libcfg4.dylib`CG4::propagate(this=0x000000010c737e90) + 1670 at CG4.cc:404
        frame #12: 0x00000001045e925a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfddb0) + 538 at OKG4Mgr.cc:88
        frame #13: 0x00000001000132da OKG4Test`main(argc=35, argv=0x00007fff5fbfde90) + 1498 at OKG4Test.cc:57
        frame #14: 0x00007fff8c89b5fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 4
    frame #4: 0x0000000105291d59 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000110943920) + 249 at G4SteppingManager2.cc:172
       169         continue;
       170       }   // NULL means the process is inactivated by a user on fly.
       171  
    -> 172       physIntLength = fCurrentProcess->
       173                       PostStepGPIL( *fTrack,
       174                                                   fPreviousStepSize,
       175                                                        &fCondition );
    (lldb) p fCurrentProcess
    (G4VProcess *) $0 = 0x00000001109c53d0
    (lldb) p fCurrentProcess->theProcessName
    (G4String) $1 = (std::__1::string = "OpBoundary")
    (lldb) 

    (lldb) f 1
    frame #1: 0x0000000105b32b17 libG4processes.dylib`G4VProcess::ResetNumberOfInteractionLengthLeft(this=0x00000001109c53d0) + 23 at G4VProcess.cc:97
       94   
       95   void G4VProcess::ResetNumberOfInteractionLengthLeft()
       96   {
    -> 97     theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
       98     theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 
       99   }
       100  




Debugging Idea
----------------

* common logging format for both simulations, so can just diff it 


Auto-interleave ?
-------------------

Redirect OptiX/CUDA logging to file ?
---------------------------------------

* https://stackoverflow.com/questions/21238303/redirecting-cuda-printf-to-a-c-stream

::

    simon:opticks blyth$ opticks-find rdbuf
    ./openmeshrap/MTool.cc:         cout_redirect out_(coutbuf.rdbuf());
    ./openmeshrap/MTool.cc:         cerr_redirect err_(cerrbuf.rdbuf()); 
    ./boostrap/BDirect.hh:        : old( std::cout.rdbuf( new_buffer ) ) 
    ./boostrap/BDirect.hh:        std::cout.rdbuf( old );
    ./boostrap/BDirect.hh:        : old( std::cerr.rdbuf( new_buffer ) ) 
    ./boostrap/BDirect.hh:        std::cerr.rdbuf( old );
    simon:opticks blyth$ 





First look at the 6 maligned
--------------------------------


::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 


::

    In [20]: ab.dumpline(range(1220,1240))
          0   1220 :                                        TO BT BT SA                                        TO BT BT SA 
          1   1221 :                                        TO BT BT SA                                        TO BT BT SA 
          2   1222 :                                        TO BT BT SA                                        TO BT BT SA 
          3   1223 :                                        TO BT BT SA                                        TO BT BT SA 
          4   1224 :                                        TO BT BT SA                                        TO BT BT SA 
          5   1225 :                                        TO BT BT SA                                        TO BT BT SA 
          6   1226 :                                        TO BT BT SA                                        TO BT BT SA 
          7   1227 :                                        TO BT BT SA                                        TO BT BT SA 
          8   1228 :                                        TO BT BT SA                                        TO BT BT SA 
          9   1229 :                                        TO BT BT SA                                        TO BT BT SA 
         10   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
         11   1231 :                                        TO BT BT SA                                        TO BT BT SA 
         12   1232 :                                        TO BT BT SA                                        TO BT BT SA 
         13   1233 :                                        TO BT BT SA                                        TO BT BT SA 
         14   1234 :                                        TO BT BT SA                                        TO BT BT SA 
         15   1235 :                                        TO BT BT SA                                        TO BT BT SA 
         16   1236 :                                        TO BT BT SA                                        TO BT BT SA 
         17   1237 :                                        TO BT BT SA                                        TO BT BT SA 
         18   1238 :                                        TO BT BT SA                                        TO BT BT SA 
         19   1239 :                                           TO BR SA                                           TO BR SA 




1230 : could be reflectivity edge

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD   




::

    In [9]: ab.recline([1230,1230])
    Out[9]: '   1230   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA '


    In [18]: a.rpolw_(slice(0,8))[1230]
    Out[18]: 
    A()sliced
    A([    [ 0.    , -1.    ,  0.    , -0.1575],    TO
           [ 0.    ,  1.    ,  0.    , -0.1575],    BR
           [-0.1969, -0.9528, -0.2283, -0.1575],    SC
           [-0.685 , -0.7165,  0.1417, -0.1575],    BT
           [-0.685 ,  0.7165, -0.1417, -0.1575],    BR
           [-0.1732,  0.9528,  0.252 , -0.1575],
           [-0.1732,  0.9528,  0.252 , -0.1575],
           [-1.    , -1.    , -1.    , -1.    ]], dtype=float32)

    In [19]: b.rpolw_(slice(0,8))[1230]
    Out[19]: 
    A()sliced
    A([    [ 0.    , -1.    ,  0.    , -0.1575],   TO
           [ 0.    ,  1.    ,  0.    , -0.1575],   BR
           [-0.1969, -0.9528, -0.2283, -0.1575],   SC
           [-0.685 , -0.7165,  0.1417, -0.1575],   BT
           [-0.685 ,  0.7165, -0.1417, -0.1575],   BR
           [-0.315 ,  0.9449, -0.0551, -0.1575],
           [-0.3307,  0.937 , -0.1024, -0.1575],
           [-0.3307,  0.937 , -0.1024, -0.1575]], dtype=float32)





Maligned Six
---------------

::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 



Manually interleaving RNG consumption logging for 1230.

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD    



    2017-12-12 19:03:34.161 INFO  [146287] [CInputPhotonSource::GeneratePrimaryVertex@163] CInputPhotonSource::GeneratePrimaryVertex n 1
    2017-12-12 19:03:34.161 ERROR [146287] [CRandomEngine::pretrack@258] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 1230 mask.size 1
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[0] :    1   1  :  0.00111702  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[1] :    2   2  :  0.502647  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[2] :    3   3  :  0.601504  :  OpAbsorption;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[3] :    4   4  :  0.938713  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[0] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.000   -0.000    1.000)  
    //                                                                     .NewMomentum :  (   0.000    0.000    0.000)  
    //                                                                      /TransCoeff :  0.938471  
    //                                                                              /_u :  0.938713  
    //                                                                       /_transmit : False 
    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[0] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (   0.000    0.000   -1.000)  
    //                                                                     .NewMomentum :  (   0.000    0.000   -1.000)  


    2017-12-12 19:03:35.820 ERROR [146287] [OPropagator::launch@183] LAUNCH NOW
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:  0.00111702492 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.5026473403   scattering_length(s.material1.z):        1000000 scattering_distance:    687866.4375 
    propagate_to_boundary  u_absorption:   0.6015041471   absorption_length(s.material1.y):       10000000 absorption_distance:      5083218.5 
    propagate_at_boundary  u_reflect:       0.93871  reflect:1   TransCoeff:   0.93847 






    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[4] :    5   1  :  0.753801  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[5] :    6   2  :  0.999847  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[6] :    7   3  :  0.43802  :  OpAbsorption;   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1 
    propagate_to_boundary  u_boundary_burn:    0.753801465 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.9998467565   scattering_length(s.material1.z):        1000000 scattering_distance:    153.2552795 
    propagate_to_boundary  u_absorption:   0.4380195737   absorption_length(s.material1.y):       10000000 absorption_distance:        8254917 



    2017-12-12 19:03:34.663 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.663 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[7] :    8   1  :  0.753801  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[8] :    9   2  :  0.999847  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[9] :   10   3  :  0.43802  :  OpAbsorption;   


    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[10] :   11   4  :  0.714032  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[11] :   12   5  :  0.330404  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[12] :   13   6  :  0.570742  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[13] :   14   7  :  0.375909  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[14] :   15   8  :  0.784978  :  OpRayleigh;   

    rayleigh_scatter_align p.direction (0 0 -1) 
    rayleigh_scatter_align p.polarization (-0 1 -0) 
    rayleigh_scatter_align.do u0:0.714032 u1:0.330404 u2:0.570742 u3:0.375909 u4:0.784978 
    rayleigh_scatter_align.do constant        (0.301043) 
    rayleigh_scatter_align.do newDirection    (0.632086 -0.301043 0.714032) 
    rayleigh_scatter_align.do newPolarization (-0.199541 -0.953611 -0.225411) 
    rayleigh_scatter_align.do doCosTheta -0.953611 doCosTheta2 0.909373   looping 0   


    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[15] :   16   1  :  0.892654  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[16] :   17   2  :  0.441063  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[17] :   18   3  :  0.773742  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[18] :   19   4  :  0.556839  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   


    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:2 
    propagate_to_boundary  u_boundary_burn:   0.8926543593 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4410631955   scattering_length(s.material1.z):        1000000 scattering_distance:     818567.125 
    propagate_to_boundary  u_absorption:   0.7737424374   absorption_length(s.material1.y):       10000000 absorption_distance:     2565162.25 
    propagate_at_boundary  u_reflect:       0.55684  reflect:0   TransCoeff:   0.88430 


    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[1] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (   0.632   -0.301    0.714)  
    //                                                                     .NewMomentum :  (   0.000    0.000   -1.000)  
    //                                                                      /TransCoeff :  0.884304  
    //                                                                              /_u :  0.556839  
    //                                                                       /_transmit : True 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[1] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (   0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (   0.381   -0.181    0.907)  







    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[19] :   20   1  :  0.775349  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[20] :   21   2  :  0.752141  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[21] :   22   3  :  0.412002  :  OpAbsorption;   



    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3 
    propagate_to_boundary  u_boundary_burn:    0.775349319 speed:    165.0280609 
    propagate_to_boundary  u_scattering:   0.7521412373   scattering_length(s.material1.z):        1000000 scattering_distance:    284831.1562 
    propagate_to_boundary  u_absorption:   0.4120023847   absorption_length(s.material1.y):        1000000 absorption_distance:     886726.125 
    propagate_at_boundary  u_reflect:       0.28246  reflect:1   TransCoeff:   0.00000  c2c2:   -1.3552 tir:1  pos (  150.0000   -77.6576    24.3052)   
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ WHATS THIS ??? DOES TIR CONSUME DIFFERENT ?



    In [7]: a.rpost_(slice(0,8))[1230]
    Out[7]: 
    A()sliced
    A([    [ -37.8781,   11.8231, -449.8989,    0.2002],    TO 
           [ -37.8781,   11.8231,  -99.9944,    1.3672],    BR   0
           [ -37.8781,   11.8231, -253.2548,    1.8781],    SC   1
           [  97.7921,  -52.7844,  -99.9944,    2.5941],    BT   2

           [ 149.9984,  -77.6556,   24.307 ,    3.4248],    BR   3   (point before was TIR)

           [ 118.2039,  -92.7959,   99.9944,    3.9308],   *BT*      << OK/G4 BT/BR
           [-191.6203, -240.3581,  449.9952,    5.566 ],   *SA*
           [   0.    ,    0.    ,    0.    ,    0.    ]])


    In [8]: b.rpost_(slice(0,8))[1230]
    Out[8]: 
    A()sliced
    A([    [ -37.8781,   11.8231, -449.8989,    0.2002],   TO
           [ -37.8781,   11.8231,  -99.9944,    1.3672],   BR 
           [ -37.8781,   11.8231, -253.2548,    1.8781],   SC
           [  97.7921,  -52.7844,  -99.9944,    2.5941],   BT
           [ 149.9984,  -77.6556,   24.307 ,    3.4248],   BR
           [ 118.2039,  -92.7959,   99.9944,    3.9308],  *BR* 
           [  34.2032, -132.8074,  -99.9944,    5.2675],  *BT*
           [-275.6348, -280.3696, -449.9952,    6.9027]]) *SA* 







    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:4 
    propagate_to_boundary  u_boundary_burn:   0.4324976802 speed:    165.0280609 
    propagate_to_boundary  u_scattering:   0.9078488946   scattering_length(s.material1.z):        1000000 scattering_distance:    96677.32812 
    propagate_to_boundary  u_absorption:   0.9121392369   absorption_length(s.material1.y):        1000000 absorption_distance:      91962.625 
    propagate_at_boundary  u_reflect:       0.20181  reflect:0   TransCoeff:   0.88556  c2c2:    0.5098 tir:0  pos (  118.2061   -92.8001   100.0000)   
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:5 
    propagate_to_boundary  u_boundary_burn:   0.7953493595 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4842039943   scattering_length(s.material1.z):        1000000 scattering_distance:         725249 
    propagate_to_boundary  u_absorption:  0.09354860336   absorption_length(s.material1.y):       10000000 absorption_distance:       23692742 
    propagate_at_surface   u_surface:       0.7505 
    propagate_at_surface   u_surface_burn:       0.9462 
    2017-12-12 19:32:41.223 ERROR [157506] [OPropagator::launch@185] LAUNCH DONE




















    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[2] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181    0.907)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[22] :   23   1  :  0.282463  :  OpBoundary;    <<< off-by-1
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[23] :   24   2  :  0.432498  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[24] :   25   3  :  0.907849  :  OpAbsorption;   

    2017-12-12 19:03:34.795 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.795 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3

    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[25] :   26   1  :  0.282463  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[26] :   27   2  :  0.432498  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[27] :   28   3  :  0.907849  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[28] :   29   4  :  0.912139  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[2] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                      /TransCoeff :  0.885559  
    //                                                                              /_u :  0.912139  
    //                                                                       /_transmit : False 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[3] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181   -0.907)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[29] :   30   1  :  0.201809  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[30] :   31   2  :  0.795349  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[31] :   32   3  :  0.484204  :  OpAbsorption;   
    2017-12-12 19:03:34.855 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.855 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[32] :   33   1  :  0.201809  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[33] :   34   2  :  0.795349  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[34] :   35   3  :  0.484204  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[35] :   36   4  :  0.0935486  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[3] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                      /TransCoeff :  0.874921  
    //                                                                              /_u :  0.0935486  
    //                                                                       /_transmit : True 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[4] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.632   -0.301   -0.714)  
    //                                                                     .NewMomentum :  (  -0.632   -0.301   -0.714)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[36] :   37   1  :  0.750533  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[37] :   38   2  :  0.946246  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[38] :   39   3  :  0.357591  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[39] :   40   4  :  0.166174  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[40] :   41   5  :  0.628917  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1242   

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[5] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.632   -0.301   -0.714)  
    //                                                                     .NewMomentum :  (  -0.632   -0.301   -0.714)  
    2017-12-12 19:03:34.926 INFO  [146287] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1

