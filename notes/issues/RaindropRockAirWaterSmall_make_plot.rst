RaindropRockAirWaterSmall_make_plot
======================================



For illustration purposes it helps to use input photons 
within the simtrace plane so add input_photons name "UpXZ1000_f8.npy"

::

    In [2]: cuss(a.seq[:,0])
    Out[2]: 
    CUSS([['w0', '                   TO BT BT SA', '           36045', '            8805'],
          ['w1', '                      TO BR SA', '            2237', '             599'],
          ['w2', '                TO BT BR BT SA', '          576461', '             503'],
          ['w3', '             TO BT BR BR BT SA', '         9223117', '              43'],
          ['w4', '                      TO BT AB', '            1229', '              28'],
          ['w5', '          TO BT BR BR BR BT SA', '       147569613', '              13'],
          ['w6', '       TO BT BR BR BR BR BT SA', '      2361113549', '               2'],
          ['w7', '                TO BT SC BT SA', '          575181', '               2'],
          ['w8', '                   TO BT BR AB', '           19405', '               2'],
          ['w9', '          TO BT BR BR BR BR AB', '        79412173', '               1'],
          ['w10', '                TO BT BT SC SA', '          552141', '               1'],
          ['w11', '                      TO SC SA', '            2157', '               1']], dtype=object)

::

    In [3]: w5
    Out[3]: array([ 142, 3305, 3339, 3451, 3474, 3922, 6281, 6795, 7237, 7516, 8521, 9358, 9719])

::

    epsilon:issues blyth$ gx
    /Users/blyth/opticks/g4cx
    epsilon:g4cx blyth$ gx ; PIDX=142 ./gxt.sh 



    In [4]: a.record[w5,0,0]
    Out[4]: 
    array([[-38.662,  28.287, -99.   ,   0.   ],
           [ 34.02 , -33.343, -99.   ,   0.   ],
           [ 18.038,  45.336, -99.   ,   0.   ],
           [-41.82 ,  22.947, -99.   ,   0.   ],
           [-26.552, -23.051, -99.   ,   0.   ],
           [ -8.587, -38.941, -99.   ,   0.   ],
           [-17.089, -45.711, -99.   ,   0.   ],
           [ 41.805, -15.498, -99.   ,   0.   ],
           [ 16.744,  45.667, -99.   ,   0.   ],
           [-38.983, -29.055, -99.   ,   0.   ],
           [-27.737, -38.445, -99.   ,   0.   ],
           [ 45.595, -13.386, -99.   ,   0.   ],
           [-40.295,  18.016, -99.   ,   0.   ]], dtype=float32)

    In [5]: a.record[w6,0,0]
    Out[5]: 
    array([[-43.501,  19.577, -99.   ,   0.   ],
           [ 46.132,  15.809, -99.   ,   0.   ]], dtype=float32)




High stats 1M : A-B
----------------------


::

    In [4]: wm = np.where( A.t.view('|S64') == B.t2.view('|S64') )[0] 
    In [5]: wm                                                                                                                                                  
    Out[5]: 
    array([     0,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11,     12,     13,     14,     15, ..., 999984, 999985, 999986, 999987, 999988, 999989, 999990,
           999991, 999992, 999993, 999994, 999995, 999996, 999997, 999998, 999999])

    In [6]: len(wm)                                                                                                                                             
    Out[6]: 999978

    In [7]: len(we)                                                                                                                                             
    Out[7]: 22

    In [10]: len(wm)                                                                                                                                            
    Out[10]: 999978

    In [11]: s_wm = a.seq[wm,0]                                                                                                                                 

    In [12]: o_wm = cuss(s_wm, wm)                                                                                                                              

    In [13]: print(o_wm)                                                                                                                                        
    [['w0' '                   TO BT BT SA' '           36045' '          885127']
     ['w1' '                      TO BR SA' '            2237' '           59974']
     ['w2' '                TO BT BR BT SA' '          576461' '           46257']
     ['w3' '             TO BT BR BR BT SA' '         9223117' '            4725']
     ['w4' '                      TO BT AB' '            1229' '            2180']
     ['w5' '          TO BT BR BR BR BT SA' '       147569613' '             947']
     ['w6' '       TO BT BR BR BR BR BT SA' '      2361113549' '             218']
     ['w7' '                TO BT SC BT SA' '          575181' '             188']
     ['w8' '                   TO BT BR AB' '           19405' '             107']
     ['w9' '    TO BT BR BR BR BR BR BT SA' '     37777816525' '              71']
     ['w10' '                      TO SC SA' '            2157' '              45']
     ['w11' '                TO BT BT SC SA' '          552141' '              28']
     ['w12' ' TO BT BR BR BR BR BR BR BT SA' '    604445064141' '              24']
     ['w13' '             TO BT BR SC BT SA' '         9202637' '              12']
     ['w14' '                TO BT BR BR AB' '          310221' '              11']
     ['w15' '                TO SC BT BT SA' '          576621' '               9']
     ['w16' '          TO BT BT SC BT BT SA' '       147614925' '               9']
     ['w17' '             TO BT SC BR BT SA' '         9221837' '               8']
     ['w18' '                         TO AB' '              77' '               7']
     ['w19' ' TO BT BR BR BR BR BR BR BR BT' '    875028003789' '               6']
     ['w20' '             TO BT BR BT SC SA' '         8833997' '               4']
     ['w21' '                   TO BR SC SA' '           34493' '               4']
     ['w22' '             TO BT BR BR BR AB' '         4963277' '               3']
     ['w23' ' TO BT BR BR BR BR BR BR BR BR' '    806308527053' '               2']
     ['w24' ' TO BT BR SC BR BR BR BR BR BT' '    875027983309' '               1']
     ['w25' '             TO SC BT BR BT SA' '         9223277' '               1']
     ['w26' '          TO BT BR BR BR BR AB' '        79412173' '               1']
     ['w27' '          TO BT SC BR BR BT SA' '       147568333' '               1']
     ['w28' '       TO BT BR SC BR BR BT SA' '      2361093069' '               1']
     ['w29' '                   TO SC BR SA' '           35693' '               1']
     ['w30' '                   TO BT BT AB' '           19661' '               1']
     ['w31' '    TO BT BR BR BR BR BR BR AB' '     20329511885' '               1']
     ['w32' '    TO BT SC BR BR BR BR BT SA' '     37777815245' '               1']
     ['w33' '    TO BT BT SC BT BR BR BT SA' '     37777861837' '               1']
     ['w34' '                      TO BR AB' '            1213' '               1']
     ['w35' '             TO BT BT SC BR SA' '         9137357' '               1']]

    In [14]:                        

::

    In [20]: a.record[:,0,1,3] = 1.    


    In [29]: np.abs(a.record[wm] - b.record[wm]).max()                                                                                                          
    Out[29]: 0.018722534

    In [29]: np.abs(a.record[wm] - b.record[wm]).max()                                                                                                          
    Out[29]: 0.018722534

    In [30]: np.where( np.abs(a.record[wm] - b.record[wm]) > 0.01 )                                                                                             
    Out[30]: 
    (array([ 18157, 125121, 467717, 499537, 624529, 695184, 759091, 779861, 938053]),
     array([1, 1, 3, 1, 1, 2, 4, 1, 1]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([2, 2, 2, 2, 2, 0, 2, 2, 2]))

    In [31]: dv_wm = np.unique(np.where( np.abs(a.record[wm] - b.record[wm]) > 0.01 )[0])                                                                       

    In [32]: dv_wm                                                                                                                                              
    Out[32]: array([ 18157, 125121, 467717, 499537, 624529, 695184, 759091, 779861, 938053])

    In [33]: seqhis_(a.seq[dv_wm,0] )                                                                                                                           
    Out[33]: 
    ['TO AB',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA']

    In [34]: wm[dv_wm]                                                                                                                                          
    Out[34]: array([ 18157, 125124, 467729, 499549, 624543, 695198, 759108, 779878, 938075])

    In [35]: dv_wm = wm[dv_wm]                                                                                                                                  

    In [36]: seqhis_(a.seq[dv_wm,0])                                                                                                                            
    Out[36]: 
    ['TO AB',
     'TO AB',
     'TO BT BT AB',
     'TO AB',
     'TO AB',
     'TO BR AB',
     'TO SC BT BT SA',
     'TO AB',
     'TO AB']

    In [37]:                                  



    In [46]: a.record[wm][dv_wm_]                                                                                                                               
    Out[46]: array([-82.325, -38.217,  64.72 , -72.788, -82.925, -37.115,   8.126, -75.773, -82.925], dtype=float32)

    In [47]: b.record[wm][dv_wm_]                                                                                                                               
    Out[47]: array([-82.311, -38.203,  64.733, -72.774, -82.907, -37.13 ,   8.115, -75.754, -82.907], dtype=float32)



Look at the 22/1M 
----------------------

All have scatters in the water::

    In [7]: seqhis_(a.seq[we,0])
    Out[7]: 
    ['TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT BR SC BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT BR SC BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT BR SC BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR',
     'TO BT SC BR BR BR BR BR BR BR']





::

    In [1]: len(wm)
    Out[1]: 999978

    In [2]: len(we)
    Out[2]: 22

    In [3]: we
    Out[3]: array([ 41595, 114799, 125032, 158475, 174993, 243023, 244904, 301474, 345307, 394971, 424120, 467407, 564111, 575295, 745197, 753378, 757835, 828015, 853528, 865287, 895530, 914361])

    In [4]: AB(we[0])
    Out[4]: 
    A : /tmp/blyth/opticks/RaindropRockAirWaterSmall/G4CXSimulateTest/ALL
    B : /tmp/blyth/opticks/RaindropRockAirWaterSmall/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL
    A(41595) : TO BT SC BR BR BR BR BR BR BR          B(41595) : TO BT SC BR BR BR BR BR BR BR
           A.t : (1000000, 64)                               B.t : (1000000, 64)
          A.t2 : (1000000, 64)                              B.t2 : (1000000, 64)
           A.n : (1000000,)                                  B.n : (1000000,)
          A.ts : (1000000, 13, 29)                          B.ts : (1000000, 13, 29)
          A.fs : (1000000, 13, 29)                          B.fs : (1000000, 13, 29)
         A.ts2 : (1000000, 13, 29)                         B.ts2 : (1000000, 13, 29)
     0 :     0.6518 :  1 :     to_sci                  0 :     0.6518 :  3 : ScintDiscreteReset :
     1 :     0.3244 :  2 :     to_bnd                  1 :     0.3244 :  4 : BoundaryDiscreteReset :
     2 :     0.2309 :  3 :     to_sca                  2 :     0.2309 :  5 : RayleighDiscreteReset :
     3 :     0.7327 :  4 :     to_abs                  3 :     0.7327 :  6 : AbsorptionDiscreteReset :
     4 :     0.1133 :  5 : at_burn_sf_sd               4 :     0.1133 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :
     5 :     0.6275 :  6 :     at_ref                  5 :     0.6275 :  8 : BoundaryDiDiTransCoeff :

     6 :     0.4789 :  1 :     to_sci                  6 :     0.4789 :  3 : ScintDiscreteReset :
     7 :     0.6990 :  2 :     to_bnd                  7 :     0.6990 :  4 : BoundaryDiscreteReset :
     8 :     0.9998 :  3 :     to_sca                  8 :     0.9998 :  5 : RayleighDiscreteReset :
     9 :     0.0067 :  4 :     to_abs                  9 :     0.0067 :  6 : AbsorptionDiscreteReset :
    10 :     0.1797 :  8 :         sc                 10 :     0.1797 : 10 : RayleighScatter :
    11 :     0.0088 :  8 :         sc                 11 :     0.0088 : 10 : RayleighScatter :
    12 :     0.5316 :  8 :         sc                 12 :     0.5316 : 10 : RayleighScatter :
    13 :     0.8436 :  8 :         sc                 13 :     0.8436 : 10 : RayleighScatter :
    14 :     0.4477 :  8 :         sc                 14 :     0.4477 : 10 : RayleighScatter :

    15 :     0.4004 :  1 :     to_sci                 15 :     0.4004 :  3 : ScintDiscreteReset :
    16 :     0.8328 :  2 :     to_bnd                 16 :     0.8328 :  4 : BoundaryDiscreteReset :
    17 :     0.0016 :  3 :     to_sca                 17 :     0.0016 :  5 : RayleighDiscreteReset :
    18 :     0.2059 :  4 :     to_abs                 18 :     0.2059 :  6 : AbsorptionDiscreteReset :
    19 :     0.5832 :  5 : at_burn_sf_sd              19 :     0.5832 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :
    20 :     0.7585 :  6 :     at_ref
                                                      20 :     0.7585 :  3 : ScintDiscreteReset :
    21 :     0.6396 :  1 :     to_sci                 21 :     0.6396 :  4 : BoundaryDiscreteReset :
    22 :     0.9837 :  2 :     to_bnd                 22 :     0.9837 :  5 : RayleighDiscreteReset :
    23 :     0.9417 :  3 :     to_sca                 23 :     0.9417 :  6 : AbsorptionDiscreteReset :
    24 :     0.2058 :  4 :     to_abs


Loose alignement at 20: TO BT SC BR* BR BR BR BR BR BR 

* expected BoundaryDiDiTransCoeff to align with at_ref 
* TODO: run these in debugger to see why do not get the expected consumption 


AB(we[1]) same again::

    15 :     0.0663 :  1 :     to_sci                  15 :     0.0663 :  3 : ScintDiscreteReset :
    16 :     0.2567 :  2 :     to_bnd                  16 :     0.2567 :  4 : BoundaryDiscreteReset :
    17 :     0.3927 :  3 :     to_sca                  17 :     0.3927 :  5 : RayleighDiscreteReset :
    18 :     0.8733 :  4 :     to_abs                  18 :     0.8733 :  6 : AbsorptionDiscreteReset :
    19 :     0.7510 :  5 : at_burn_sf_sd               19 :     0.7510 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :
    20 :     0.8006 :  6 :     at_ref
                                                       20 :     0.8006 :  3 : ScintDiscreteReset :
    21 :     0.7021 :  1 :     to_sci                  21 :     0.7021 :  4 : BoundaryDiscreteReset :
    22 :     0.5178 :  2 :     to_bnd                  22 :     0.5178 :  5 : RayleighDiscreteReset :
    23 :     0.5743 :  3 :     to_sca                  23 :     0.5743 :  6 : AbsorptionDiscreteReset :
    24 :     0.9636 :  4 :     to_abs

AB(we[2]) same, alignment lost at first BR following SC::

    18 :     0.2799 :  8 :         sc                  18 :     0.2799 : 10 : RayleighScatter :
    19 :     0.2901 :  8 :         sc                  19 :     0.2901 : 10 : RayleighScatter :

    20 :     0.6490 :  1 :     to_sci                  20 :     0.6490 :  3 : ScintDiscreteReset :
    21 :     0.8381 :  2 :     to_bnd                  21 :     0.8381 :  4 : BoundaryDiscreteReset :
    22 :     0.0342 :  3 :     to_sca                  22 :     0.0342 :  5 : RayleighDiscreteReset :
    23 :     0.3264 :  4 :     to_abs                  23 :     0.3264 :  6 : AbsorptionDiscreteReset :
    24 :     0.6856 :  5 : at_burn_sf_sd               24 :     0.6856 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :
    25 :     0.5878 :  6 :     at_ref
                                                       25 :     0.5878 :  3 : ScintDiscreteReset :
    26 :     0.5236 :  1 :     to_sci                  26 :     0.5236 :  4 : BoundaryDiscreteReset :
    27 :     0.9741 :  2 :     to_bnd                  27 :     0.9741 :  5 : RayleighDiscreteReset :

AB(we[8])::

    22 :     0.2898 :  8 :         sc                  22 :     0.2898 : 10 : RayleighScatter :
    23 :     0.9960 :  8 :         sc                  23 :     0.9960 : 10 : RayleighScatter :
    24 :     0.1025 :  8 :         sc                  24 :     0.1025 : 10 : RayleighScatter :

    25 :     0.7586 :  1 :     to_sci                  25 :     0.7586 :  3 : ScintDiscreteReset :
    26 :     0.5196 :  2 :     to_bnd                  26 :     0.5196 :  4 : BoundaryDiscreteReset :
    27 :     0.0803 :  3 :     to_sca                  27 :     0.0803 :  5 : RayleighDiscreteReset :
    28 :     0.4960 :  4 :     to_abs                  28 :     0.4960 :  6 : AbsorptionDiscreteReset :
    29 :     0.2536 :  5 : at_burn_sf_sd               29 :     0.2536 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :
    30 :     0.7670 :  6 :     at_ref
                                                       30 :     0.7670 :  3 : ScintDiscreteReset :
    31 :     0.9490 :  1 :     to_sci                  31 :     0.9490 :  4 : BoundaryDiscreteReset :
    32 :     0.9607 :  2 :     to_bnd                  32 :     0.9607 :  5 : RayleighDiscreteReset :


::

     787 #ifdef DEBUG_TAG
     788     const float u_boundary_burn = curand_uniform(&rng) ;  // needed for random consumption alignment with Geant4 G4OpBoundaryProcess::PostStepDoIt
     789 #endif
     790     const float u_reflect = curand_uniform(&rng) ;
     791     bool reflect = u_reflect > TransCoeff  ;
     792 
     793 #ifdef DEBUG_TAG
     794     stagr& tagr = ctx.tagr ;
     795     tagr.add( stag_at_burn_sf_sd, u_boundary_burn);
     796     tagr.add( stag_at_ref,  u_reflect);
     797 #endif
     798 
     799 #ifdef DEBUG_PIDX
     800     if(ctx.idx == base->pidx)
     801     {
     802     printf("//qsim.propagate_at_boundary idx %d u_boundary_burn %10.4f u_reflect %10.4f TransCoeff %10.4f reflect %d \n",
     803               ctx.idx,  u_boundary_burn, u_reflect, TransCoeff, reflect  );
     804     }
     805 #endif
     806 







::

    171 void U4Process::ClearNumberOfInteractionLengthLeft(const G4Track& aTrack, const G4Step& aStep)
    172 {
    173     G4ProcessManager* mgr = GetManager();
    174     G4ProcessVector* procv = mgr->GetProcessList() ;
    175     for(int i=0 ; i < procv->entries() ; i++)
    176     {
    177         G4VProcess* proc = (*procv)[i] ;
    178         unsigned type = ProcessType(Name(proc)) ;
    179         if(IsNormalProcess(type))
    180         {
    181             G4VDiscreteProcess* dproc = dynamic_cast<G4VDiscreteProcess*>(proc) ;
    182             assert(dproc);
    183             dproc->G4VDiscreteProcess::PostStepDoIt( aTrack, aStep );
    184         }
    185     }
    186 }

    069 inline bool U4Process::IsNormalProcess(unsigned type)
     70 {
     71     return type == U4Process_OpRayleigh || type == U4Process_OpAbsorption ;
     72 }




rejig paths continued
------------------------

::

    2022-07-17 03:46:24.152 INFO  [346371] [U4Recorder::EndOfEventAction@83] 
    2022-07-17 03:46:24.153 INFO  [346371] [U4Recorder::EndOfRunAction@81] 
    2022-07-17 03:46:24.156 INFO  [346371] [main@207] outdir [/tmp/blyth/opticks/J000/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL]
    2022-07-17 03:46:24.156 INFO  [346371] [main@208]  desc [ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL]
    U4Random::saveProblemIdx m_problem_idx.size 0 ()
    2022-07-17 03:46:24.176 INFO  [346371] [main@213] outdir /tmp/blyth/opticks/J000/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL
    === ./u4s.sh : logdir /tmp/blyth/opticks/U4RecorderTest
    N[blyth@localhost u4]$ 
    N[blyth@localhost u4]$ 


Path inconsistency::

    epsilon:u4 blyth$ ./u4s.sh grab 
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                       TMP_GEOMDIR : /tmp/blyth/opticks/J000 
                           GEOMDIR : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo 

                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON_.sh
                              GEOM : J000
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_FRAME : Hama:0:1000
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy
        OPTICKS_INPUT_PHOTON_LABEL : DownXZ1000
                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON.sh 
                         ScriptDir : ./../bin 
              OPTICKS_INPUT_PHOTON : DownXZ1000_f8.npy 
        OPTICKS_INPUT_PHOTON_FRAME : Hama:0:1000 
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/DownXZ1000_f8.npy 

    === ./u4s.sh : grab FOLD /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL
    BASH_SOURCE                    : ./../bin/rsync.sh 
    xdir                           : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL/ 
    from                           : P:/Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL/ 
    to                             : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL/ 
    receiving incremental file list
    rsync: change_dir "/Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL" failed: No such file or directory (2)

    sent 79 bytes  received 248 bytes  43.60 bytes/sec
    total size is 0  speedup is 0.00
    rsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1679) [Receiver=3.1.3]
    rsync: [Receiver] write error: Broken pipe (32)
    == ./../bin/rsync.sh tto /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL jpg mp4 npy
    epsilon:u4 blyth$ 


