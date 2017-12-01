sc_ab_re_alignment
======================

Not so easy to --scattercheat --absorbcheat ?
--------------------------------------------------------

Cheating reflection (removing the randomness over whether to reflect/transmit) 
so that the bi-simulation can stay point-by-point aligned was straightforward 
as the code that does that is tightly contained in DsG4OpBoundaryProcess.

Not so easy to play this trick with SC/AB/RE because there is play off 
between all three or None of them.  This might be tractable in Opticks, 
but it is much more difficuult with G4 as the decision 
over which process to go with is happening in the ProcessManager guts of G4 
(see below).


Think in terms of propagate_to_boundary outcomes (exclude reemission for simplicity):

* sail to boundary (boundary_distance)
* scattered before boundary  (scattering_distance)
* absorbed before boundary (absorption_distance)


Having control over the "GetMeanFreePath" perhaps not sufficient to 
game the system : as another random throw is done to convert that length 
into a distance.  

Could per-step switch off via DBL_MAX-ing 



::

   g4-;g4-cls G4OpRayleigh    // G4VDiscreteProcess
   g4-;g4-cls G4OpAbsorption  // G4VDiscreteProcess



    In [15]: -np.log(np.linspace(0,1,101))
    Out[15]: 
    array([    inf,  4.6052,  3.912 ,  3.5066,  3.2189,  2.9957,  2.8134,  2.6593,  2.5257,  2.4079,  2.3026,  2.2073,  2.1203,  2.0402,  1.9661,  1.8971,  1.8326,  1.772 ,  1.7148,  1.6607,  1.6094,
            1.5606,  1.5141,  1.4697,  1.4271,  1.3863,  1.3471,  1.3093,  1.273 ,  1.2379,  1.204 ,  1.1712,  1.1394,  1.1087,  1.0788,  1.0498,  1.0217,  0.9943,  0.9676,  0.9416,  0.9163,  0.8916,
            0.8675,  0.844 ,  0.821 ,  0.7985,  0.7765,  0.755 ,  0.734 ,  0.7133,  0.6931,  0.6733,  0.6539,  0.6349,  0.6162,  0.5978,  0.5798,  0.5621,  0.5447,  0.5276,  0.5108,  0.4943,  0.478 ,
            0.462 ,  0.4463,  0.4308,  0.4155,  0.4005,  0.3857,  0.3711,  0.3567,  0.3425,  0.3285,  0.3147,  0.3011,  0.2877,  0.2744,  0.2614,  0.2485,  0.2357,  0.2231,  0.2107,  0.1985,  0.1863,
            0.1744,  0.1625,  0.1508,  0.1393,  0.1278,  0.1165,  0.1054,  0.0943,  0.0834,  0.0726,  0.0619,  0.0513,  0.0408,  0.0305,  0.0202,  0.0101, -0.    ])



::

     59 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     60 {
     61     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     62     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     63 
     64     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
     65     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
     66 
     67     if (absorption_distance <= scattering_distance)
     68     {
     69         if (absorption_distance <= s.distance_to_boundary)
     70         {

::

    GetMeanFreePath method is equivalent to the interpolated texture lookup result 
    which is in s.material1.z (scattering_length) 

    273 G4double DsG4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,
    274                                      G4double ,
    275                                      G4ForceCondition* )




Perhaps could cheat by returning DBL_MAX from disfavored ?
-------------------------------------------------------------


::

    071 G4double G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(
     72                              const G4Track& track,
     73                  G4double   previousStepSize,
     74                  G4ForceCondition* condition
     75                 )
     76 {               
     77   if ( (previousStepSize < 0.0) || (theNumberOfInteractionLengthLeft<=0.0)) {
     78     // beggining of tracking (or just after DoIt of this process)
     79     ResetNumberOfInteractionLengthLeft();
     80   } else if ( previousStepSize > 0.0) {
     81     // subtract NumberOfInteractionLengthLeft 
     82     SubtractNumberOfInteractionLengthLeft(previousStepSize);
     83   } else {       
     84     // zero step             
     85     //  DO NOTHING 
     86   }
     87       
     88   // condition is set to "Not Forced"
     89   *condition = NotForced;
     90                 
     91   // get mean free path
     92   currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
     93       
     94   G4double value;
     95   if (currentInteractionLength <DBL_MAX) {
     96     value = theNumberOfInteractionLengthLeft * currentInteractionLength;
     97   } else {
     98     value = DBL_MAX;
     99   }              
    100 #ifdef G4VERBOSE 
    101   if (verboseLevel>1){
    102     G4cout << "G4VDiscreteProcess::PostStepGetPhysicalInteractionLength ";
    103     G4cout << "[ " << GetProcessName() << "]" <<G4endl;
    104     track.GetDynamicParticle()->DumpInfo(); 
    105     G4cout << " in Material  " <<  track.GetMaterial()->GetName() <<G4endl;
    106     G4cout << "InteractionLength= " << value/cm <<"[cm] " <<G4endl;
    107   }
    108 #endif
    109   return value;
    110 }




Whos gonna call GetMeanFreePath ? G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
--------------------------------------------------------------------------------------------

::

    simon:cfg4 blyth$ tboolean-;tboolean-box --okg4 -D


    (lldb) b "G4OpRayleigh::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 
    Breakpoint 1: 2 locations.
    (lldb) b "G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 
    Breakpoint 2: where = libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) + 4 [inlined] G4Track::GetMaterial() const at G4OpAbsorption.cc:127, address = 0x0000000105a81c34
    (lldb) b "DsG4OpRayleigh::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 
    Breakpoint 3: where = libcfg4.dylib`DsG4OpRayleigh::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) + 29 at DsG4OpRayleigh.cc:277, address = 0x00000001043566dd
    (lldb) b "DsG4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 
    Breakpoint 4: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.
    (lldb) 


::

    (lldb) bt
    * thread #1: tid = 0xf0825, 0x0000000105a81c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) [inlined] G4Track::GetMaterial(this=<unavailable>) const at G4Track.icc:153, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
      * frame #0: 0x0000000105a81c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) [inlined] G4Track::GetMaterial(this=<unavailable>) const at G4Track.icc:153
        frame #1: 0x0000000105a81c34 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(this=0x000000010d971960, aTrack=0x0000000134034830, (null)=<unavailable>, (null)=0x000000010d8ec9d8) + 4 at G4OpAbsorption.cc:127
        frame #2: 0x0000000105a7e490 libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=0x000000010d971960, track=0x0000000134034830, previousStepSize=<unavailable>, condition=0x000000010d8ec9d8) + 112 at G4VDiscreteProcess.cc:92
        frame #3: 0x00000001051ddd67 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength() [inlined] G4VProcess::PostStepGPIL(this=0x000000010d971960, track=<unavailable>, previousStepSize=<unavailable>, condition=<unavailable>) + 14 at G4VProcess.hh:503
        frame #4: 0x00000001051ddd59 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x000000010d8ec850) + 249 at G4SteppingManager2.cc:172
        frame #5: 0x00000001051dc73e libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010d8ec850) + 366 at G4SteppingManager.cc:180
        frame #6: 0x00000001051e6771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010d8ec810, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #7: 0x000000010513e727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010d8ec780, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #8: 0x00000001050c0611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010c6e1230, i_event=0) + 49 at G4RunManager.cc:399
        frame #9: 0x00000001050c04db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010c6e1230, n_event=10, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #10: 0x00000001050bf913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010c6e1230, n_event=10, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #11: 0x0000000104434946 libcfg4.dylib`CG4::propagate(this=0x000000010c6e1040) + 1670 at CG4.cc:352
        frame #12: 0x000000010453525a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfdee0) + 538 at OKG4Mgr.cc:88
        frame #13: 0x00000001000132da OKG4Test`main(argc=29, argv=0x00007fff5fbfdfc0) + 1498 at OKG4Test.cc:57
        frame #14: 0x00007fff92a345fd libdyld.dylib`start + 1
    (lldb) 



::

    simon:cfg4 blyth$ g4-;g4-cls G4OpAbsorption
    simon:cfg4 blyth$ g4-;g4-cls G4OpRayleigh
    simon:cfg4 blyth$ g4-;g4-cls G4SteppingManager2
    simon:cfg4 blyth$ g4-;g4-cls G4VDiscreteProcess



DONE : avoid accidental history alignment deviation
------------------------------------------------------

* NB not using --reflectcheat in the below but still mostly aligned because testauto simplifies things by 
  using the perfectSpecularSurface for the object and perfectAbsorbSurface for the container : 
  so there is little leeway for randomness to creep in 

* BUT, an accidentally history aligned "TO SC SA" photon is causing deviation fails across the board 

* any possibility of --scattercheat ?


::

    In [4]: ab.rpost_dv.dvs[2].av
    Out[4]: 
    A()sliced
    A([[[  19.3382,   -8.7951, -449.9127,    0.2002],
            [-272.5792,  214.9775,   82.2665,    2.3579],
            [-410.259 ,  449.9952,  241.3904,    3.4101]]])

    In [5]: ab.rpost_dv.dvs[2].bv
    Out[5]: 
    A()sliced
    A([[[  19.3382,   -8.7951, -449.9127,    0.2002],
            [  -5.423 ,   10.199 , -404.7535,    0.3833],
            [-338.5218,  449.9952, -301.91  ,    2.2553]]])




::

    simon:opticks blyth$ tboolean-;tboolean-box --okg4 --testauto


    [2017-12-01 19:00:32,466] p94255 {/Users/blyth/opticks/ana/tboolean.py:27} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171201-1900 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171201-1900 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         3.85/5 =  0.77  (pval:0.570 prob:0.430)  
    0000               8d    390593    390548             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO SA
    0001              8ad    208878    208867             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO SR SA
    0002              86d       368       417             3.06        0.882 +- 0.046        1.133 +- 0.055  [3 ] TO SC SA
    0003             86ad        62        70             0.48        0.886 +- 0.112        1.129 +- 0.135  [4 ] TO SR SC SA
    0004             8a6d        38        38             0.00        1.000 +- 0.162        1.000 +- 0.162  [4 ] TO SC SR SA
    0005               4d        38        43             0.31        0.884 +- 0.143        1.132 +- 0.173  [2 ] TO AB
    0006            8a6ad        16        10             0.00        1.600 +- 0.400        0.625 +- 0.198  [5 ] TO SR SC SR SA
    0007              4ad         6         7             0.00        0.857 +- 0.350        1.167 +- 0.441  [3 ] TO SR AB
    0008            866ad         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SR SC SC SA
    .                             600000    600000         3.85/5 =  0.77  (pval:0.570 prob:0.430)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         3.37/4 =  0.84  (pval:0.497 prob:0.503)  
    0000             1080    390593    390548             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO|SA
    0001             1280    208878    208867             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO|SR|SA
    0002             10a0       368       417             3.06        0.882 +- 0.046        1.133 +- 0.055  [3 ] TO|SA|SC
    0003             12a0       117       118             0.00        0.992 +- 0.092        1.009 +- 0.093  [4 ] TO|SR|SA|SC
    0004             1008        38        43             0.31        0.884 +- 0.143        1.132 +- 0.173  [2 ] TO|AB
    0005             1208         6         7             0.00        0.857 +- 0.350        1.167 +- 0.441  [3 ] TO|SR|AB
    .                             600000    600000         3.37/4 =  0.84  (pval:0.497 prob:0.503)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.62/3 =  0.21  (pval:0.891 prob:0.109)  
    0000               12    390593    390548             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] Vm Rk
    0001              122    209246    209284             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] Vm Vm Rk
    0002             1222       100       108             0.31        0.926 +- 0.093        1.080 +- 0.104  [4 ] Vm Vm Vm Rk
    0003               22        38        43             0.31        0.884 +- 0.143        1.132 +- 0.173  [2 ] Vm Vm
    0004            12222        17        10             0.00        1.700 +- 0.412        0.588 +- 0.186  [5 ] Vm Vm Vm Vm Rk
    0005              222         6         7             0.00        0.857 +- 0.350        1.167 +- 0.441  [3 ] Vm Vm Vm
    .                             600000    600000         0.62/3 =  0.21  (pval:0.891 prob:0.109)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 6d0343f9434d01eb932af1e1cb716bbd 22ea8cb862c05bf8fb67471c291e77dc  600000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}

    rpost_dv maxdvmax:543.30036317 maxdv:[0.013763847773702764, 0.013763847773702764, 543.3003631702627] 
     0000            :                          TO SA :  390593   390548  :    390191 3121528/   1456: 0.000  mx/mn/av 0.01376/     0/6.159e-06  eps:0.0002    
     0001            :                       TO SR SA :  208878   208867  :    208696 2504352/    390: 0.000  mx/mn/av 0.01376/     0/1.834e-06  eps:0.0002    

     0002            :                       TO SC SA :     368      417  :         1      12/      7: 0.583  mx/mn/av  543.3/     0/ 131.4  eps:0.0002    

    rpol_dv maxdvmax:1.51181101799 maxdv:[0.007874011993408203, 0.0, 1.5118110179901123] 
     0000            :                          TO SA :  390593   390548  :    390191 2341146/      2: 0.000  mx/mn/av 0.007874/     0/6.727e-09  eps:0.0002    
     0001            :                       TO SR SA :  208878   208867  :    208696 1878264/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    

     0002            :                       TO SC SA :     368      417  :         1       9/      6: 0.667  mx/mn/av  1.512/     0/0.7052  eps:0.0002    

    ox_dv maxdvmax:543.313659668 maxdv:[0.000152587890625, 9.1552734375e-05, 543.3136596679688] 
     0000            :                          TO SA :  390593   390548  :    390191 6243056/      0: 0.000  mx/mn/av 0.0001526/     0/2.651e-06  eps:0.0002    
     0001            :                       TO SR SA :  208878   208867  :    208696 3339136/      0: 0.000  mx/mn/av 9.155e-05/     0/1.408e-06  eps:0.0002    

     0002            :                       TO SC SA :     368      417  :         1      16/      9: 0.562  mx/mn/av  543.3/     0/ 38.74  eps:0.0002    

    c2p : {'seqmat_ana': 0.20745893850775779, 'pflags_ana': 0.84359450816083359, 'seqhis_ana': 0.77099423966857661} c2pmax: 0.843594508161  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 1.5118110179901123, 'rpost_dv': 543.3003631702627} rmxs_max_: 543.30036317  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': 543.3136596679688} pmxs_max_: 543.313659668  CUT ok.pdvmax 0.001  RC:99 




