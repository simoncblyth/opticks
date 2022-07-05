gxs_ab_hama_body_log
=======================

gxs_ab.sh::

     14 source ../bin/GEOM_.sh
     15 FOLD_MODE=GXS source ../bin/AB_FOLD.sh
     16 
     17 ${IPYTHON:-ipython} --pdb -i tests/G4CXSimulateTest_ab.py $*

First look, lots out of history alignment::

    In [6]: w = np.where( a.seq[:,0] != b.seq[:,0])[0] ; len(w)
    Out[6]: 2139

Quite a lot aligned too, too much to be accidental::

    In [11]: wm = np.where( a.seq[:,0] == b.seq[:,0])[0] ; len(wm)
    Out[11]: 7861


Maybe need microstep skipping (or skipping virtual skins) like did previously.

Histories of first 10::

    In [9]: seqhis_(a.seq[:10,0])
    Out[9]: 
    ['TO BT BT BT BR BT BT BT SA',
     'TO BT BT AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA']

    In [10]: seqhis_(b.seq[:10,0])
    Out[10]: 
    ['TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA']

2/TO BT BT [BT] BT BT SA/history matched but time off from mid-point/probably degenerate surfaces mean using wrong groupvel::

    In [21]: a.record[2,:7] - b.record[2,:7]
    Out[21]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[-0.   ,  0.   ,  0.   ,  0.301],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[-0.004,  0.002,  0.   ,  0.302],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)


point-to-point position time deltas within A and B::

    In [24]: a.record[2,1:7,0] - a.record[2,0:6,0]
    Out[24]: 
    array([[  0.   ,   0.   , 806.775,   3.728],
           [  0.   ,   0.   ,   5.   ,   0.025],
           [  0.   ,   0.   , 178.225,   *0.896*],
           [  0.   ,   0.   , 184.558,   0.616],
           [  0.071,  -0.044,   5.002,   0.025],
           [  9.177,  -5.715, 810.44 ,   3.746]], dtype=float32)

    In [25]: b.record[2,1:7,0] - b.record[2,0:6,0]
    Out[25]: 
    array([[  0.   ,   0.   , 806.775,   3.728],
           [  0.   ,   0.   ,   5.   ,   0.025],
           [  0.   ,   0.   , 178.225,   *0.594*],
           [  0.   ,   0.   , 184.558,   0.616],
           [  0.071,  -0.044,   5.002,   0.025],
           [  9.181,  -5.717, 810.44 ,   3.745]], dtype=float32)


4/TO BT BT [BT] BT BT SA/history matched but time off from mid-point::

    In [20]: a.record[4,:7] - b.record[4,:7]
    Out[20]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],  ## time off from middle point TO BT BT [BT] BT BT SA
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.   ,  0.301],
            [ 0.   ,  0.   , -0.   ,  0.   ],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],
            [ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.013,  0.014,  0.   ,  0.303],
            [ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)


5/TO AB::

    In [18]: a.record[5,:2] - b.record[5,:2]
    Out[18]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.003, -0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)



Checking those with matched histories shows no BR on internal layers in first 100 anyhow::

    In [14]: seqhis_( b.seq[wm[:100],0] )
    Out[14]: 
    ['TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',




Scripted interleaving with sysrap/ABR.py
-------------------------------------------

DONE: script such interleaving "AB(0)" and move the result : BT/BR/... alongside the decision random

* sysrap/ABR.py presents repr of two objects side-by-side 

Developed with the fully aligned raindrop geom::

    In [2]: AB(4)
    Out[2]: 
    A(4) : TO BT BT SA                                                                      B(4) : TO BT BT SA                                                            
           A.t : (10000, 48)                                                                       B.t : (10000, 48)                                                      
           A.n : (10000,)                                                                          B.n : (10000,)                                                         
          A.ts : (10000, 10, 29)                                                                  B.ts : (10000, 10, 29)                                                  
          A.fs : (10000, 10, 29)                                                                  B.fs : (10000, 10, 29)                                                  
         A.ts2 : (10000, 10, 29)                                                                 B.ts2 : (10000, 10, 29)                                                  
     0 :     0.9251 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            0 :     0.9251 :  3 : ScintDiscreteReset :                                   
     1 :     0.0530 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            1 :     0.0530 :  4 : BoundaryDiscreteReset :                                
     2 :     0.1631 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             2 :     0.1631 :  5 : RayleighDiscreteReset :                                
     3 :     0.8897 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             3 :     0.8897 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.5666 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                 4 :     0.5666 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.2414 :  6 :     at_ref : u_reflect > TransCoeff                               5 :     0.2414 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
     6 :     0.4937 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            6 :     0.4937 :  3 : ScintDiscreteReset :                                   
     7 :     0.3212 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            7 :     0.3212 :  4 : BoundaryDiscreteReset :                                
     8 :     0.0786 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             8 :     0.0786 :  5 : RayleighDiscreteReset :                                
     9 :     0.1479 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             9 :     0.1479 :  6 : AbsorptionDiscreteReset :                              
    10 :     0.5987 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                10 :     0.5987 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    11 :     0.4265 :  6 :     at_ref : u_reflect > TransCoeff                              11 :     0.4265 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    12 :     0.2435 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           12 :     0.2435 :  3 : ScintDiscreteReset :                                   
    13 :     0.4892 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           13 :     0.4892 :  4 : BoundaryDiscreteReset :                                
    14 :     0.4095 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            14 :     0.4095 :  5 : RayleighDiscreteReset :                                
    15 :     0.6676 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            15 :     0.6676 :  6 : AbsorptionDiscreteReset :                              
    16 :     0.6269 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                16 :     0.6269 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    17 :     0.2769 :  7 :    sf_burn : qsim::propagate_at_surface burn                     17 :     0.2769 :  9 : AbsorptionEffDetect :                                  
    18 :     0.0000 :  0 :      undef : undef                                               18 :     0.0000 :  0 : Unclassified :                                         
    19 :     0.0000 :  0 :      undef : undef                                               19 :     0.0000 :  0 : Unclassified :                                         


Normally there is one less consumption clump than there are step points. But when there is a BR 
there is an extra consumption clump from the Geant4 StepTooSmall and Opticks mimicking that with burns to retain alignment::

    In [5]: AB(3)
    Out[5]: 
    A(3) : TO BR SA                                                                         B(3) : TO BR SA                                                               
           A.t : (10000, 48)                                                                       B.t : (10000, 48)                                                      
           A.n : (10000,)                                                                          B.n : (10000,)                                                         
          A.ts : (10000, 10, 29)                                                                  B.ts : (10000, 10, 29)                                                  
          A.fs : (10000, 10, 29)                                                                  B.fs : (10000, 10, 29)                                                  
         A.ts2 : (10000, 10, 29)                                                                 B.ts2 : (10000, 10, 29)                                                  
     0 :     0.9690 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            0 :     0.9690 :  3 : ScintDiscreteReset :                                   
     1 :     0.4947 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            1 :     0.4947 :  4 : BoundaryDiscreteReset :                                
     2 :     0.6734 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             2 :     0.6734 :  5 : RayleighDiscreteReset :                                
     3 :     0.5628 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             3 :     0.5628 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.1202 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                 4 :     0.1202 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.9765 :  6 :     at_ref : u_reflect > TransCoeff                               5 :     0.9765 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
     6 :     0.1358 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            6 :     0.1358 :  3 : ScintDiscreteReset :                                   
     7 :     0.5890 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            7 :     0.5890 :  4 : BoundaryDiscreteReset :                                
     8 :     0.4906 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             8 :     0.4906 :  5 : RayleighDiscreteReset :                                
     9 :     0.3284 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             9 :     0.3284 :  6 : AbsorptionDiscreteReset :                              
                                                                                                                                                                          
    10 :     0.9114 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           10 :     0.9114 :  3 : ScintDiscreteReset :                                   
    11 :     0.1907 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           11 :     0.1907 :  4 : BoundaryDiscreteReset :                                
    12 :     0.9637 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            12 :     0.9637 :  5 : RayleighDiscreteReset :                                
    13 :     0.8976 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            13 :     0.8976 :  6 : AbsorptionDiscreteReset :                              
    14 :     0.6243 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                14 :     0.6243 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    15 :     0.7102 :  7 :    sf_burn : qsim::propagate_at_surface burn                     15 :     0.7102 :  9 : AbsorptionEffDetect :                                  
    16 :     0.0000 :  0 :      undef : undef                                               16 :     0.0000 :  0 : Unclassified :                                         
    17 :     0.0000 :  0 :      undef : undef                                               17 :     0.0000 :  0 : Unclassified :          


    In [8]: AB(36)
    Out[8]: 
    A(36) : TO BT BR BT SA                                                                  B(36) : TO BT BR BT SA                                                        
           A.t : (10000, 48)                                                                       B.t : (10000, 48)                                                      
           A.n : (10000,)                                                                          B.n : (10000,)                                                         
          A.ts : (10000, 10, 29)                                                                  B.ts : (10000, 10, 29)                                                  
          A.fs : (10000, 10, 29)                                                                  B.fs : (10000, 10, 29)                                                  
         A.ts2 : (10000, 10, 29)                                                                 B.ts2 : (10000, 10, 29)                                                  
     0 :     0.2405 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            0 :     0.2405 :  3 : ScintDiscreteReset :                                   
     1 :     0.4503 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            1 :     0.4503 :  4 : BoundaryDiscreteReset :                                
     2 :     0.2029 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             2 :     0.2029 :  5 : RayleighDiscreteReset :                                
     3 :     0.5092 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             3 :     0.5092 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.2154 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                 4 :     0.2154 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.1141 :  6 :     at_ref : u_reflect > TransCoeff                               5 :     0.1141 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
     6 :     0.3870 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            6 :     0.3870 :  3 : ScintDiscreteReset :                                   
     7 :     0.8183 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            7 :     0.8183 :  4 : BoundaryDiscreteReset :                                
     8 :     0.2030 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             8 :     0.2030 :  5 : RayleighDiscreteReset :                                
     9 :     0.7006 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             9 :     0.7006 :  6 : AbsorptionDiscreteReset :                              
    10 :     0.5327 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                10 :     0.5327 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    11 :     0.9862 :  6 :     at_ref : u_reflect > TransCoeff                              11 :     0.9862 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    12 :     0.5105 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           12 :     0.5105 :  3 : ScintDiscreteReset :                                   
    13 :     0.3583 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           13 :     0.3583 :  4 : BoundaryDiscreteReset :                                
    14 :     0.9380 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            14 :     0.9380 :  5 : RayleighDiscreteReset :                                
    15 :     0.4586 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            15 :     0.4586 :  6 : AbsorptionDiscreteReset :                              
                                                                                                                                                                          
    16 :     0.9189 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           16 :     0.9189 :  3 : ScintDiscreteReset :                                   
    17 :     0.1870 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           17 :     0.1870 :  4 : BoundaryDiscreteReset :                                
    18 :     0.2109 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            18 :     0.2109 :  5 : RayleighDiscreteReset :                                
    19 :     0.9003 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            19 :     0.9003 :  6 : AbsorptionDiscreteReset :                              
    20 :     0.0704 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                20 :     0.0704 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    21 :     0.7765 :  6 :     at_ref : u_reflect > TransCoeff                              21 :     0.7765 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    22 :     0.3422 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           22 :     0.3422 :  3 : ScintDiscreteReset :                                   
    23 :     0.1178 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           23 :     0.1178 :  4 : BoundaryDiscreteReset :                                
    24 :     0.5520 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            24 :     0.5520 :  5 : RayleighDiscreteReset :                                
    25 :     0.3090 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            25 :     0.3090 :  6 : AbsorptionDiscreteReset :                              
    26 :     0.0165 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                26 :     0.0165 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    27 :     0.4159 :  7 :    sf_burn : qsim::propagate_at_surface burn                     27 :     0.4159 :  9 : AbsorptionEffDetect :                                  
    28 :     0.0000 :  0 :      undef : undef                                               28 :     0.0000 :  0 : Unclassified :                                         
    29 :     0.0000 :  0 :      undef : undef                                               29 :     0.0000 :  0 : Unclassified :                                         




Manually interleaving A(0) B(0) shows where alignment is lost
---------------------------------------------------------------

::

    In [29]: A(0)
    Out[29]: 
    A(0) : TO BT BT BT BR BT BT BT SA
           A.t : (10000, 48) 
           A.n : (10000,) 
          A.ts : (10000, 9, 29) 
          A.fs : (10000, 9, 29) 
         A.ts2 : (10000, 9, 29) 

    B(0) : TO BT BT BT BT BT SA
           B.t : (10000, 48) 
           B.n : (10000,) 
          B.ts : (10000, 10, 29) 
          B.fs : (10000, 10, 29) 
         B.ts2 : (10000, 10, 29) 


     0 :     0.7402 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     1 :     0.4385 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     2 :     0.5170 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     3 :     0.1570 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
     4 :     0.0714 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
     5 :     0.4625 :  6 :     at_ref : u_reflect > TransCoeff 

     0 :     0.7402 :  3 : ScintDiscreteReset :  
     1 :     0.4385 :  4 : BoundaryDiscreteReset :  
     2 :     0.5170 :  5 : RayleighDiscreteReset :  
     3 :     0.1570 :  6 : AbsorptionDiscreteReset :  
     4 :     0.0714 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
     5 :     0.4625 :  8 : BoundaryDiDiTransCoeff :  



     6 :     0.2276 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     7 :     0.3294 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     8 :     0.1441 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     9 :     0.1878 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    10 :     0.9154 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    11 :     0.5401 :  6 :     at_ref : u_reflect > TransCoeff 

     6 :     0.2276 :  3 : ScintDiscreteReset :  
     7 :     0.3294 :  4 : BoundaryDiscreteReset :  
     8 :     0.1441 :  5 : RayleighDiscreteReset :  
     9 :     0.1878 :  6 : AbsorptionDiscreteReset :  
    10 :     0.9154 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
    11 :     0.5401 :  8 : BoundaryDiDiTransCoeff :  



    12 :     0.9747 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    13 :     0.5475 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    14 :     0.6532 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    15 :     0.2302 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    16 :     0.3389 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    17 :     0.7614 :  6 :     at_ref : u_reflect > TransCoeff 

    12 :     0.9747 :  3 : ScintDiscreteReset :  
    13 :     0.5475 :  4 : BoundaryDiscreteReset :  
    14 :     0.6532 :  5 : RayleighDiscreteReset :  
    15 :     0.2302 :  6 : AbsorptionDiscreteReset :  

    ##  ALIGNMENT LOST HERE : THATS MAYBE A StepTooSmall ?


    18 :     0.5457 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    19 :     0.9703 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    20 :     0.2112 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    21 :     0.9469 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    22 :     0.5530 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    23 :     0.9776 :  6 :     at_ref : u_reflect > TransCoeff 


    16 :     0.3389 :  3 : ScintDiscreteReset :  
    17 :     0.7614 :  4 : BoundaryDiscreteReset :  
    18 :     0.5457 :  5 : RayleighDiscreteReset :  
    19 :     0.9703 :  6 : AbsorptionDiscreteReset :  
    20 :     0.2112 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
    21 :     0.9469 :  8 : BoundaryDiDiTransCoeff :  





TODO: get gxr working to visualize this
-------------------------------------------

 
