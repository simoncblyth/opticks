AB_SC_Position_Time_mismatch
===============================

Masked run on 1st "TO AB"
-----------------------------


Find some indices for masked running::

    In [55]: ab.a.dindex("TO AB")
    Out[55]: '--dindex=37922,61642,92906'

    In [56]: ab.b.dindex("TO AB")
    Out[56]: '--dindex=37922,61642,92906'

::

   tboolean-;tboolean-box --okg4 --align --mask 37922 --pindex 0 -DD 

   tboolean-;tboolean-box --okg4 --align 







AFTER LOG DOUBLE FIX AB POSITIONS MATCHING
---------------------------------------------

::

    In [3]: ab.aselhis = "TO AB"

    In [4]: ab.a.rpost()
    Out[4]: 
    A()sliced
    A([[[  32.3038,  -30.831 , -449.8989,    0.2002],
            [  32.3038,  -30.831 , -380.7631,    0.4309]],

           [[ -14.9751,   25.2704, -449.8989,    0.2002],
            [ -14.9751,   25.2704, -282.4066,    0.7587]],

           [[ -32.0422,    6.9507, -449.8989,    0.2002],
            [ -32.0422,    6.9507, -223.9929,    0.9534]]])

    In [5]: ab.b.rpost()
    Out[5]: 
    A()sliced
    A([[[  32.3038,  -30.831 , -449.8989,    0.2002],
            [  32.3038,  -30.831 , -380.7631,    0.4309]],

           [[ -14.9751,   25.2704, -449.8989,    0.2002],
            [ -14.9751,   25.2704, -282.4066,    0.7587]],

           [[ -32.0422,    6.9507, -449.8989,    0.2002],
            [ -32.0422,    6.9507, -223.9929,    0.9534]]])

    In [6]: ab.a.rpost() - ab.b.rpost()
    Out[6]: 
    A()sliced
    A([[[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]]])






::

    2017-12-12 13:08:52.888 ERROR [39772] [OPropagator::launch@183] LAUNCH NOW
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:   0.8371831775 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4740084112   scattering_length(s.material1.z):        1000000 scattering_distance:    746530.1875 
    propagate_to_boundary  u_absorption:   0.9999930859   absorption_length(s.material1.y):       10000000 absorption_distance:    69.14162445 
    2017-12-12 13:08:52.902 ERROR [39772] [OPropagator::launch@185] LAUNCH DONE

::

     60 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     61 {
     62     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     63     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     64 
     65 #ifdef WITH_ALIGN_DEV
     66     float u_boundary_burn = curand_uniform(&rng) ;
     67     float u_scattering = curand_uniform(&rng) ;
     68     float u_absorption = curand_uniform(&rng) ;
     69 
     70     float scattering_distance = -s.material1.z*log(double(u_scattering)) ;   // .z:scattering_length
     71     float absorption_distance = -s.material1.y*log(double(u_absorption)) ;   // .y:absorption_length 
     72     //  see notes/issues/AB_SC_Position_Time_mismatch.rst
     73 #else
     74     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
     75     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
     76 #endif
     77 
     78 #ifdef WITH_ALIGN_DEV_DEBUG
     79     rtPrintf("propagate_to_boundary  u_boundary_burn:%15.10g speed:%15.10g \n", u_boundary_burn, speed );
     80     rtPrintf("propagate_to_boundary  u_scattering:%15.10g   scattering_length(s.material1.z):%15.10g scattering_distance:%15.10g \n", u_scattering, s.material1.z, scattering_distance );
     81     rtPrintf("propagate_to_boundary  u_absorption:%15.10g   absorption_length(s.material1.y):%15.10g absorption_distance:%15.10g \n", u_absorption, s.material1.y, absorption_distance );
     82 #endif





EXPLAINED : difference between float/double logf/log  
-----------------------------------------------------------

::

    simon:optixrap blyth$ thrust_curand_printf 37922
    thrust_curand_printf
     i0 37922 i1 37923 q0 0 q1 16
     id:37922 thread_offset:0 seq0:0 seq1:16 
     0.837183  0.474008  0.999993  0.850346 
     0.547168  0.109779  0.189688  0.667250 
     0.225588  0.575690  0.891344  0.320829 
     0.125657  0.859036  0.326287  0.256005 
    simon:optixrap blyth$ 

::

    In [18]: rng = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))

    In [20]: rng[37922].ravel()[:10]
    Out[20]: 
    array([ 0.83718318,  0.47400841,  0.99999309,  0.85034555,  0.5471676 ,
            0.10977882,  0.18968813,  0.66725016,  0.22558783,  0.57569039])

    In [21]: u = rng[37922].ravel()[2]


    In [23]: np.log(u)
    Out[23]: -6.9141626967131213e-06

    In [24]: np.log(u)*1e7
    Out[24]: -69.141626967131216

    In [27]: np.log(np.float32(u))*1e7
    Out[27]: -69.141628955549095

    In [26]: np.log(u-1e-6)*1e7    ## taking log makes very sensitive to precision of RNG
    Out[26]: -79.141701109357243



Try the log on GPU::

    simon:thrustrap blyth$ thrust_curand_printf 37922
    thrust_curand_printf
     i0 37922 i1 37923 q0 0 q1 16 logf N
     id:37922 thread_offset:0 seq0:0 seq1:16 
     0.837183  0.474008  0.999993  0.850346 
     0.547168  0.109779  0.189688  0.667250 
     0.225588  0.575690  0.891344  0.320829 
     0.125657  0.859036  0.326287  0.256005 

    simon:thrustrap blyth$ LOGF=1 thrust_curand_printf 37922
    thrust_curand_printf
     i0 37922 i1 37923 q0 0 q1 16 logf Y
     id:37922 thread_offset:0 seq0:0 seq1:16 
     0.837183  1777123.500000      1777123.824  0.474008  7465302.000000      7465302.124  0.999993  *68.669197*      *69.14162697*  0.850346  1621124.250000      1621124.804 
     0.547168  6030000.500000      6030001.264  0.109779  22092874.000000      22092876.51  0.189688  16623739.000000       16623739.7  0.667250  4045902.250000      4045902.563 
     0.225588  14890456.000000      14890457.06  0.575690  5521852.000000      5521852.825  0.891344  1150252.375000      1150252.989  0.320829  11368461.000000      11368461.59 
     0.125657  20742028.000000      20742029.26  0.859036  1519448.875000      1519449.013  0.326287  11199765.000000      11199765.43  0.256005  13625582.000000      13625583.46 
    simon:thrustrap blyth$ 


thrap/tests/thrust_curand_printf.cu::

     36     __device__
     37     void operator()(unsigned id)
     38     {
     39         unsigned thread_offset = 0 ;
     40         curandState s;
     41         curand_init(_seed, id + thread_offset, _offset, &s);
     42         printf(" id:%4u thread_offset:%u seq0:%llu seq1:%llu \n", id, thread_offset, _seq0, _seq1 );
     43 
     44         for(T i = _zero ; i < _seq1 ; ++i)
     45         {
     46             float f = curand_uniform(&s);
     47             if( i < _seq0 ) continue ;
     48 
     49             printf(" %lf ", f );
     50 
     51             if(_logf)
     52             {
     53                 float lf = -logf(f)*1e7f ;
     54                 printf(" %lf ", lf );
     55 
     56                 //double d(f) ;   
     57                 //double ld = -log(d)*1e7 ; 
     58 
     59                 //double ld = -log(double(f))*1e7 ; 
     60                 float ld = -log(double(f))*1e7 ;
     61                 printf(" %15.10g ", ld );
     62 
     63             }



AB rpost
-----------

::

    2017-12-12 11:48:21.401 INFO  [14851] [CInputPhotonSource::GeneratePrimaryVertex@163] CInputPhotonSource::GeneratePrimaryVertex n 1
    2017-12-12 11:48:21.401 ERROR [14851] [CRandomEngine::pretrack@258] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 37922 mask.size 1
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[0] :    1   1  : 0.83718317747116089  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[1] :    2   2  : 0.47400841116905212  
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[2] :    3   3  : 0.99999308586120605  
    G4SteppingManager2_cc_181_ : Dumping lengths collected by _181 after PostStep process loop  
    //                                                  .fCurrentProcess.theProcessName :  OpBoundary  
    //                                                                   .physIntLength :  1.79769e+308  
    //                                                  .fCurrentProcess.theProcessName :  OpRayleigh  
    //                                                                   .physIntLength :  746530  
    //                                                  .fCurrentProcess.theProcessName :  OpAbsorption  
    //                                                                   .physIntLength :  69.1416  


    In [51]: fpb = ab.b.ox[:,0] - ab.b.so[:,0] ; fpb 
    Out[52]: 
    A([    [   0.    ,    0.    ,  *69.1416*,   0.2306],
           [   0.    ,    0.    ,  167.4904,    0.5587],
           [   0.    ,    0.    ,  225.9042,    0.7535]], dtype=float32)

    In [49]: fpa = ab.a.ox[:,0] - ab.a.so[:,0] ; fpa
    Out[50]: 
    A([    [   0.    ,    0.    ,  *68.6692*,   0.2291],
           [   0.    ,    0.    ,  167.0023,    0.5571],
           [   0.    ,    0.    ,  225.4427,    0.752 ]], dtype=float32)


    2017-12-12 12:25:24.055 ERROR [26173] [OPropagator::launch@183] LAUNCH NOW
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:   0.8371831775 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4740084112   scattering_length(s.material1.z):        1000000 scattering_distance:    746530.1875 
    propagate_to_boundary  u_absorption:   0.9999930859   absorption_length(s.material1.y):       10000000 absorption_distance:    68.66919708 
    2017-12-12 12:25:24.069 ERROR [26173] [OPropagator::launch@185] LAUNCH DONE





    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  1.79769e+308  

    //                                opticks.ana.cfg4lldb.G4Transportation_cc_517_.[0] : AlongStepGetPhysicalInteractionLength Exit  
    //                                                                             this : G4Transportation_cc_517 
    //                                                                   /startPosition :  (  32.306  -30.833 -449.900)  
    //                                                                /startMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                                       /newSafety :  0.100006  
    //                                                            .fGeometryLimitedStep : False 
    //                                                              .fFirstStepInVolume : True 
    //                                                               .fLastStepInVolume : False 
    //                                                                .fMomentumChanged : False 
    //                                                          .fShortStepOptimisation : False 
    //                                                           .fTransportEndPosition :  (  32.306  -30.833 -380.758)  
    //                                                        .fTransportEndMomentumDir :  (  -0.000   -0.000    1.000)  
    //                                                               .fEndPointDistance :  69.1416  
    //                                               .fParticleChange.thePositionChange :  (   0.000    0.000    0.000)  
    //                                      .fParticleChange.theMomentumDirectionChange :  (   0.000    0.000    0.000)  
    //                                               .fLinearNavigator.fNumberZeroSteps :  0  
    //                                               .fLinearNavigator.fLastStepWasZero : False 

    //                              opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_.[0] : Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL 
    //                                                                             this : G4SteppingManager2_cc_270 
    //                                                  .fCurrentProcess.theProcessName :  Transportation  
    //                                                                   .physIntLength :  69.1416  
    //                                                                    .PhysicalStep :  69.1416  
    //                                                                     .fStepStatus :  fPostStepDoItProc  

    //                               opticks.ana.cfg4lldb.G4TrackingManager_cc_131_.[0] : Step Conclusion : TrackingManager step loop just after Stepping()  
    //                                                                             this : G4TrackingManager 
    //                                                   .fpSteppingManager.fStepStatus :  fPostStepDoItProc  
    //                                                  .fpSteppingManager.PhysicalStep :  69.1416  
    //                                .fpSteppingManager.fCurrentProcess.theProcessName :  OpAbsorption  
    //                                .fpSteppingManager.fStep.fpPreStepPoint.fPosition :  (  32.306  -30.833 -449.900)  
    //                              .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime :  0.2  
    //                       .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection :  (  -0.000   -0.000    1.000)  
    //                               .fpSteppingManager.fStep.fpPostStepPoint.fPosition :  (  32.306  -30.833 -380.758)  
    //                             .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime :  0.430632  
    //                      .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection :  (  -0.000   -0.000    1.000)  
    //                                                       CRandomEngine_cc_flatExit_ : 3 
    //                                                         G4Transportation_cc_517_ : 1 
    //                                                        G4TrackingManager_cc_131_ : 1 
    //                                                       G4SteppingManager2_cc_270_ : 1 
    //                                                       G4SteppingManager2_cc_181_ : 0 
    2017-12-12 11:48:21.976 INFO  [14851] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1





::

    tboolean-;tboolean-box-ip

    In [9]: ab.his
    Out[9]: 
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.24/7 =  0.03  (pval:1.000 prob:0.000)  
    0000             8ccd     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5420             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       349             0.00        1.000 +- 0.054        1.000 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        31        29             0.07        1.069 +- 0.192        0.935 +- 0.174  [3 ] TO SC SA
    0005            86ccd        27        24             0.18        1.125 +- 0.217        0.889 +- 0.181  [5 ] TO BT BT SC SA
    0006          8cbbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [7 ] TO BT BR BR BR BT SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         6         7             0.00        0.857 +- 0.350        1.167 +- 0.441  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BT BT AB
    0012          8cc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0013               4d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO AB
    0014           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0015           8cb6cd         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [6 ] TO BT SC BR BT SA
    0016       8cbbbbb6cd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BT SA
    0017           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0018            8cc6d         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO SC BT BT SA
    0019          8cb6bcd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR SC BR BT SA
    .                             100000    100000         0.24/7 =  0.03  (pval:1.000 prob:0.000)  

    In [10]: ab.aselhis = "TO AB"

    In [11]: ab.a.rpost()
    Out[11]: 
    A()sliced
    A([[    [  32.3038,  -30.831 , -449.8989,    0.2002],
            [  32.3038,  -30.831 , -381.2311,    0.4291]],

           [[ -14.9751,   25.2704, -449.8989,    0.2002],
            [ -14.9751,   25.2704, -282.9021,    0.7569]],

           [[ -32.0422,    6.9507, -449.8989,    0.2002],
            [ -32.0422,    6.9507, -224.4608,    0.9522]]])

    In [12]: ab.b.rpost()
    Out[12]: 
    A()sliced
    A([[    [  32.3038,  -30.831 , -449.8989,    0.2002],
            [  32.3038,  -30.831 , -380.7631,    0.4309]],

           [[ -14.9751,   25.2704, -449.8989,    0.2002],
            [ -14.9751,   25.2704, -282.4066,    0.7587]],

           [[ -32.0422,    6.9507, -449.8989,    0.2002],
            [ -32.0422,    6.9507, -223.9929,    0.9534]]])


    In [13]: rpa = ab.a.rpost()
    In [14]: rpb = ab.b.rpost()


    In [28]: dpa = rpa[:,1] - rpa[:,0] ; dpa   ## AB-TO rpost 
    Out[28]: 
    A()sliced
    A([    [   0.    ,    0.    ,   68.6678,    0.2289],
           [   0.    ,    0.    ,  166.9968,    0.5567],
           [   0.    ,    0.    ,  225.4381,    0.752 ]])

    In [29]: dpb = rpb[:,1] - rpb[:,0] ; dpb 
    Out[29]: 
    A()sliced
    A([    [   0.    ,    0.    ,   69.1358,    0.2307],
           [   0.    ,    0.    ,  167.4923,    0.5585],
           [   0.    ,    0.    ,  225.906 ,    0.7532]])



    ## rpost is limited precision from domain compression, 
    ## so probably cannot conclude anything from velocity differences below

    In [36]: dpa[:,2]/dpa[:,3]     
    Out[36]: 
    A()sliced
    A([ 300.0052,  299.9991,  299.7942])

    In [37]: dpb[:,2]/dpb[:,3]
    Out[37]: 
    A()sliced
    A([ 299.6525,  299.9027,  299.9296])


    ## so use the non-compressed float32 initial/final photon position from so/ox
    ## shows no velocity difference between the simulations

    In [49]: fpa = ab.a.ox[:,0] - ab.a.so[:,0] ; fpa
    Out[50]: 
    A([    [   0.    ,    0.    ,   68.6692,    0.2291],
           [   0.    ,    0.    ,  167.0023,    0.5571],
           [   0.    ,    0.    ,  225.4427,    0.752 ]], dtype=float32)

    In [51]: fpb = ab.b.ox[:,0] - ab.b.so[:,0] ; fpb 
    Out[52]: 
    A([    [   0.    ,    0.    ,   69.1416,    0.2306],
           [   0.    ,    0.    ,  167.4904,    0.5587],
           [   0.    ,    0.    ,  225.9042,    0.7535]], dtype=float32)


    In [53]: fpb[:,2]/fpb[:,3]
    Out[53]: 
    A()sliced
    A([ 299.7925,  299.7924,  299.7924], dtype=float32)

    In [54]: fpa[:,2]/fpa[:,3]
    Out[54]: 
    A()sliced
    A([ 299.7924,  299.7924,  299.7924], dtype=float32)






SC rpost
--------------

* no point working on the scatter, unless can get scatter position to match

::

    tboolean-;tboolean-box-ip

    In [8]: ab.recline([595,595])
    Out[8]: '    595    595 :                                           TO SC SA                                  TO SC BT BR BT SA '


    In [4]: ab.a.rpost_(slice(0,4))[595]
    Out[4]: 
    A()sliced
    A([    [ -41.8696,   39.9977, -449.8989,    0.2002],
           [ -41.8696,   39.9977, -216.0924,    0.9796],
           [-449.9952,  207.9029, -138.5469,    2.4744],
           [   0.    ,    0.    ,    0.    ,    0.    ]])

    In [6]: ab.b.rpost_(slice(0,8))[595]
    Out[6]: 
    A()sliced
    A([    [ -41.8696,   39.9977, -449.8989,    0.2002],
           [ -41.8696,   39.9977, -216.0374,    0.9803],
           [-105.8027,   66.2867,  -99.9944,    1.4307],
           [-149.9984,   84.4825,   47.4715,    2.3701],
           [-134.2526,   90.9515,   99.9944,    2.7046],
           [  58.5514,  170.2726,  449.9952,    4.0632],
           [   0.    ,    0.    ,    0.    ,    0.    ],
           [   0.    ,    0.    ,    0.    ,    0.    ]])



History
-----------

::


    tboolean-;tboolean-box-ip

    In [9]: ab.his
    Out[9]: 
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.24/7 =  0.03  (pval:1.000 prob:0.000)  
    0000             8ccd     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5420             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       349             0.00        1.000 +- 0.054        1.000 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        31        29             0.07        1.069 +- 0.192        0.935 +- 0.174  [3 ] TO SC SA
    0005            86ccd        27        24             0.18        1.125 +- 0.217        0.889 +- 0.181  [5 ] TO BT BT SC SA
    0006          8cbbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [7 ] TO BT BR BR BR BT SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         6         7             0.00        0.857 +- 0.350        1.167 +- 0.441  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         4             0.00        0.750 +- 0.433        1.333 +- 0.667  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BT BT AB
    0012          8cc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0013               4d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO AB
    0014           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0015           8cb6cd         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [6 ] TO BT SC BR BT SA
    0016       8cbbbbb6cd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT SC BR BR BR BR BR BT SA
    0017           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0018            8cc6d         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] TO SC BT BT SA
    0019          8cb6bcd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR SC BR BT SA
    .                             100000    100000         0.24/7 =  0.03  (pval:1.000 prob:0.000)  


