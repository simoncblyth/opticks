post recording has seqmat zeros
==================================






After Fix seqmat chi2 is 0.93
---------------------------------

tconcentric.py::

    .                seqmat_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000       212.62/229 =  0.93  (pval:0.774 prob:0.226)  
       0               343231        669845       670001             0.02        1.000 +- 0.001        1.000 +- 0.001  [6 ] Gd Ac LS Ac MO Ac
       1                   11         83950        84149             0.24        0.998 +- 0.003        1.002 +- 0.003  [2 ] Gd Gd
       2              3432311         65732        64910             5.17        1.013 +- 0.004        0.987 +- 0.004  [7 ] Gd Gd Ac LS Ac MO Ac
       3               443231         28955        28718             0.97        1.008 +- 0.006        0.992 +- 0.006  [6 ] Gd Ac LS Ac MO MO
       4                 2231         23188        23170             0.01        1.001 +- 0.007        0.999 +- 0.007  [4 ] Gd Ac LS LS
       5              3443231         17716        18028             2.72        0.983 +- 0.007        1.018 +- 0.008  [7 ] Gd Ac LS Ac MO MO Ac
       6              3432231         15325        15297             0.03        1.002 +- 0.008        0.998 +- 0.008  [7 ] Gd Ac LS LS Ac MO Ac
       7             34323111         10939        10935             0.00        1.000 +- 0.010        1.000 +- 0.010  [8 ] Gd Gd Gd Ac LS Ac MO Ac
       8                  111         10576        10652             0.27        0.993 +- 0.010        1.007 +- 0.010  [3 ] Gd Gd Gd
       9      343231323443231          6955         6885             0.35        1.010 +- 0.012        0.990 +- 0.012  [15] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
      10          34323443231          6038         5990             0.19        1.008 +- 0.013        0.992 +- 0.013  [11] Gd Ac LS Ac MO MO Ac LS Ac MO Ac
      11          34323132231          4423         4472             0.27        0.989 +- 0.015        1.011 +- 0.015  [11] Gd Ac LS LS Ac Gd Ac LS Ac MO Ac
      12              4443231          3160         3198             0.23        0.988 +- 0.018        1.012 +- 0.018  [7 ] Gd Ac LS Ac MO MO MO
      13              4432311          3006         2990             0.04        1.005 +- 0.018        0.995 +- 0.018  [7 ] Gd Gd Ac LS Ac MO MO
      14            343231111          2857         2939             1.16        0.972 +- 0.018        1.029 +- 0.019  [9 ] Gd Gd Gd Gd Ac LS Ac MO Ac
      15                22311          2791         2814             0.09        0.992 +- 0.019        1.008 +- 0.019  [5 ] Gd Gd Ac LS LS
      16                 1111          2441         2425             0.05        1.007 +- 0.020        0.993 +- 0.020  [4 ] Gd Gd Gd Gd
      17             34322311          1997         1895             2.67        1.054 +- 0.024        0.949 +- 0.022  [8 ] Gd Gd Ac LS LS Ac MO Ac
      18             34322231          1844         1904             0.96        0.968 +- 0.023        1.033 +- 0.024  [8 ] Gd Ac LS LS LS Ac MO Ac
      19                22231          1789         1847             0.93        0.969 +- 0.023        1.032 +- 0.024  [5 ] Gd Ac LS LS LS
    .                               1000000      1000000       212.62/229 =  0.93  (pval:0.774 prob:0.226)  
    [2016-11-23 12:13:21,934] p10672 {/Users/blyth/opticks/ana/tconcentric.py:240} INFO - early exit as non-interactive



FIX REQUIRED matSwap BASED ON NEXT BOUNDARY
------------------------------------------------

::

    -        bool matSwap = boundary_status == StepTooSmall ; 
    +        bool matSwap = next_boundary_status == StepTooSmall ; 


post (canned) recording::


    ( 4)  BT/*DR*     LaR                                                     
    [   4](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.690 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )
           // step after DR will be skipped as STS:StepTooSmall SO NEED TO MAP_SWAP when next_boundary_status is STS 

    ( 5)  DR/*NA*     STS                                  POST_SKIP MAT_SWAP 
    [   5](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )
          // this step is (post)skipped as boundary_status is StepTooSmall




Issue
--------

Following the transition to canned rather than live recording in CRecorder tconcentric.py shows terrible seqmat chi2 due to zeros::

    .                               1000000      1000000     48279.63/273 = 176.85  (pval:0.000 prob:1.000)  
       0               343231        669845       670001             0.02        1.000 +- 0.001        1.000 +- 0.001  [6 ] Gd Ac LS Ac MO Ac
       1                   11         83950        84149             0.24        0.998 +- 0.003        1.002 +- 0.003  [2 ] Gd Gd
       2              3432311         65732        64910             5.17        1.013 +- 0.004        0.987 +- 0.004  [7 ] Gd Gd Ac LS Ac MO Ac
       3               443231         28955        28718             0.97        1.008 +- 0.006        0.992 +- 0.006  [6 ] Gd Ac LS Ac MO MO
       4                 2231         23188        23170             0.01        1.001 +- 0.007        0.999 +- 0.007  [4 ] Gd Ac LS LS
       5              3443231         17716        10318          1952.29        1.717 +- 0.013        0.582 +- 0.006  [7 ] Gd Ac LS Ac MO MO Ac
       6              3432231         15325        15298             0.02        1.002 +- 0.008        0.998 +- 0.008  [7 ] Gd Ac LS LS Ac MO Ac
       7             34323111         10939        10932             0.00        1.001 +- 0.010        0.999 +- 0.010  [8 ] Gd Gd Gd Ac LS Ac MO Ac
       8                  111         10576        10652             0.27        0.993 +- 0.010        1.007 +- 0.010  [3 ] Gd Gd Gd
       9              3343231             0         7710          7710.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Gd Ac LS Ac MO Ac Ac
      10      343231323443231          6955         1616          3325.74        4.304 +- 0.052        0.232 +- 0.006  [15] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
      11          34323443231          6038         1522          2697.65        3.967 +- 0.051        0.252 +- 0.006  [11] Gd Ac LS Ac MO MO Ac LS Ac MO Ac
      12      343231323343231             0         5269          5269.00        0.000 +- 0.000        0.000 +- 0.000  [15] Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac
      13          34323132231          4423         4470             0.25        0.989 +- 0.015        1.011 +- 0.015  [11] Gd Ac LS LS Ac Gd Ac LS Ac MO Ac
      14          34323343231             0         4469          4469.00        0.000 +- 0.000        0.000 +- 0.000  [11] Gd Ac LS Ac MO Ac Ac LS Ac MO Ac
      15              4443231          3160          815          1383.40        3.877 +- 0.069        0.258 +- 0.009  [7 ] Gd Ac LS Ac MO MO MO
      16              4432311          3006         2990             0.04        1.005 +- 0.018        0.995 +- 0.018  [7 ] Gd Gd Ac LS Ac MO MO
      17            343231111          2857         2939             1.16        0.972 +- 0.018        1.029 +- 0.019  [9 ] Gd Gd Gd Gd Ac LS Ac MO Ac
      18                22311          2791         2814             0.09        0.992 +- 0.019        1.008 +- 0.019  [5 ] Gd Gd Ac LS LS
      19                 1111          2441         2425             0.05        1.007 +- 0.020        0.993 +- 0.020  [4 ] Gd Gd Gd Gd
    .                               1000000      1000000     48279.63/273 = 176.85  (pval:0.000 prob:1.000)  


    simon:ana blyth$ tconcentric.py --dbgzero 
    /Users/blyth/opticks/ana/tconcentric.py --dbgzero
    [2016-11-22 18:09:44,008] p5447 {/Users/blyth/opticks/ana/tconcentric.py:234} INFO - tag 1 src torch det concentric c2max 2.0 ipython False 
    [2016-11-22 18:09:44,008] p5447 {/Users/blyth/opticks/ana/ab.py:78} INFO - AB.load START 
    [2016-11-22 18:09:45,194] p5447 {/Users/blyth/opticks/ana/evt.py:380} WARNING -  t :   0.000 132.000 : tot 1000000 over 83 0.000  under 0 0.000 : mi      0.100 mx    389.164  
    [2016-11-22 18:09:45,733] p5447 {/Users/blyth/opticks/ana/evt.py:483} INFO - pflags2(=seq2msk(seqhis)) and pflags  MISMATCH    num_msk_mismatch: 2 
    [2016-11-22 18:09:46,593] p5447 {/Users/blyth/opticks/ana/ab.py:94} INFO - AB.load DONE 
    [2016-11-22 18:09:46,941] p5447 {/Users/blyth/opticks/ana/ab.py:123} INFO - AB.init_point START
    [2016-11-22 18:09:47,499] p5447 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point DONE
    AB(1,torch,concentric)  None 0 
    A concentric/torch/  1 :  20161122-1502 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    B concentric/torch/ -1 :  20161122-1502 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
    ...

    .                seqmat_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000     48279.63/273 = 176.85  (pval:0.000 prob:1.000)  
       9              3343231             0         7710          7710.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Gd Ac LS Ac MO Ac Ac
      12      343231323343231             0         5269          5269.00        0.000 +- 0.000        0.000 +- 0.000  [15] Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac
      14          34323343231             0         4469          4469.00        0.000 +- 0.000        0.000 +- 0.000  [11] Gd Ac LS Ac MO Ac Ac LS Ac MO Ac
      20              4343231             0         2383          2383.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Gd Ac LS Ac MO Ac MO
      30            223343231             0         1013          1013.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Gd Ac LS Ac MO Ac Ac LS LS
      34             34343231             0          969           969.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Gd Ac LS Ac MO Ac MO Ac
      35          11323343231             0          956           956.00        0.000 +- 0.000        0.000 +- 0.000  [11] Gd Ac LS Ac MO Ac Ac LS Ac Gd Gd
      38     3432311323343231             0          755           755.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS Ac MO Ac Ac LS Ac Gd Gd Ac LS Ac MO Ac
      39             33432311             0          736           736.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Gd Gd Ac LS Ac MO Ac Ac
      40         343223343231             0          733           733.00        0.000 +- 0.000        0.000 +- 0.000  [12] Gd Ac LS Ac MO Ac Ac LS LS Ac MO Ac
      48     3432313233432311             0          547           547.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac
      52         343233432311             0          445           445.00        0.000 +- 0.000        0.000 +- 0.000  [12] Gd Gd Ac LS Ac MO Ac Ac LS Ac MO Ac
      60          44323343231             0          328           328.00        0.000 +- 0.000        0.000 +- 0.000  [11] Gd Ac LS Ac MO Ac Ac LS Ac MO MO
      66      443231323343231             0          272           272.00        0.000 +- 0.000        0.000 +- 0.000  [15] Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO MO
      68        2231323343231             0          256           256.00        0.000 +- 0.000        0.000 +- 0.000  [13] Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS LS
      74             43432311             0          229           229.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Gd Gd Ac LS Ac MO Ac MO
      76             33343231             0          203           203.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Gd Ac LS Ac MO Ac Ac Ac
      77     3343231323343231             0          200           200.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac Ac
      78             33432231             0          199           199.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Gd Ac LS LS Ac MO Ac Ac
      80     3432313223343231             0          192           192.00        0.000 +- 0.000        0.000 +- 0.000  [16] Gd Ac LS Ac MO Ac Ac LS LS Ac Gd Ac LS Ac MO Ac
    .                               1000000      1000000     48279.63/273 = 176.85  (pval:0.000 prob:1.000)  
    [2016-11-22 18:09:47,851] p5447 {/Users/blyth/opticks/ana/tconcentric.py:240} INFO - early exit as non-interactive




Cause
--------

Look into seqmat line 9, looking at history sequence within a material selection and then flipping to select on history
and look at material sequence reveals the point with the material discrepancy to be DR diffuse reflect::

    In [1]: ab.selmat = "Gd Ac LS Ac MO Ac Ac" ; ab.his
    Out[1]: 
    .                seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                     0         7710      7710.00/0 = 7710.00  (pval:nan prob:nan)  
       0              89ccccd             0         7710          7710.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT BT BT DR SA

    In [4]: ab.selhis = "TO BT BT BT BT DR SA" ; ab.mat
    Out[4]: 
    .                seqmat_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                  7540         7710     15250.00/1 = 15250.00  (pval:0.000 prob:1.000)  
       0              3343231             0         7710          7710.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Gd Ac LS Ac MO Ac Ac
       1              3443231          7540            0          7540.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Gd Ac LS Ac MO MO Ac


Selection flipping from selmat to selhis like this is good way to debug zeros::

    In [6]: ab.selmat = "Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac" ; ab.his
    Out[6]: 
    .                seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                     0         5269      5269.00/0 = 5269.00  (pval:nan prob:nan)  
       0      8cccccccc9ccccd             0         5269          5269.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA

    In [8]: ab.selhis = "TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA" ; ab.mat 
    Out[8]: 
    .                seqmat_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                  5339         5269     10608.00/1 = 10608.00  (pval:0.000 prob:1.000)  
       0      343231323443231          5339            0          5339.00        0.000 +- 0.000        0.000 +- 0.000  [15] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
       1      343231323343231             0         5269          5269.00        0.000 +- 0.000        0.000 +- 0.000  [15] Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac
    ##                                                                                                                                     ^^
    ## DR material labelled as MO with Opticks (where headed after DR)
    ## DR material labelled as Ac with CFG4 (where headed before DR)


Same again, the pattern repeats::

    In [10]: ab.selmat = "Gd Ac LS Ac MO Ac Ac LS Ac MO Ac" ; ab.his
    Out[10]: 
    .                seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                     0         4469      4469.00/0 = 4469.00  (pval:nan prob:nan)  
       0          8cccc9ccccd             0         4469          4469.00        0.000 +- 0.000        0.000 +- 0.000  [11] TO BT BT BT BT DR BT BT BT BT SA

    In [12]: ab.selhis = "TO BT BT BT BT DR BT BT BT BT SA" ; ab.mat 
    Out[12]: 
    .                seqmat_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                  4494         4469      8963.00/1 = 8963.00  (pval:0.000 prob:1.000)  
       0          34323443231          4494            0          4494.00        0.000 +- 0.000        0.000 +- 0.000  [11] Gd Ac LS Ac MO MO Ac LS Ac MO Ac
       1          34323343231             0         4469          4469.00        0.000 +- 0.000        0.000 +- 0.000  [11] Gd Ac LS Ac MO Ac Ac LS Ac MO Ac



Opticks code check shows no kludging for DR SURFACE_DREFLECT
---------------------------------------------------------------

::

    515 __device__ int
    516 propagate_at_surface(Photon &p, State &s, curandState &rng)
    517 {
    518 
    519     float u = curand_uniform(&rng);
    520 
    521     if( u < s.surface.y )   // absorb   
    522     {
    523         s.flag = SURFACE_ABSORB ;
    524         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    525         return BREAK ;
    526     }
    527     else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    528     {
    529         s.flag = SURFACE_DETECT ;
    530         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    531         return BREAK ;
    532     }
    533     else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    534     {
    535         s.flag = SURFACE_DREFLECT ;
    536         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    537         return CONTINUE;
    538     }
    539     else
    540     {
    541         s.flag = SURFACE_SREFLECT ;
    542         propagate_at_specular_reflector(p, s, rng );
    543         return CONTINUE;
    544     }
    545 }


CFG4 Dumping of mismatched photons
-------------------------------------

Find some record_id to dump::

    In [21]: ab.selhis = "TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA" ; ab.mat 
    Out[21]: 
    .                seqmat_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                  5339         5269     10608.00/1 = 10608.00  (pval:0.000 prob:1.000)  
       0      343231323443231          5339            0          5339.00        0.000 +- 0.000        0.000 +- 0.000  [15] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
       0      8cccccccc9ccccd          5339         5269             0.46        1.013 +- 0.014        0.987 +- 0.014  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
       1      343231323343231             0         5269          5269.00        0.000 +- 0.000        0.000 +- 0.000  [15] Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac

    In [23]: ab.b.psel_dindex(limit=10)
    Out[23]: '--dindex=97,359,363,453,1267,1276,1298,1468,1812,1859'

    In [24]: ab.b.psel_dindex(limit=10,reverse=True)
    Out[24]: '--dindex=999969,999931,999504,999373,999215,999211,998990,998889,998747,998692'

    In [25]: ab.his
    Out[25]: 
    .                seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                                  5339         5269         0.46/0 =  0.46  (pval:nan prob:nan)  
       0      8cccccccc9ccccd          5339         5269             0.46        1.013 +- 0.014        0.987 +- 0.014  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA



Run the simulations::

    tconcentric-tt --dindex=97,359,363,453,1267,1276,1298,1468,1812,1859


::

    2016-11-22 21:15:03.881 INFO  [1732801] [CRecorder::dump@1234] CRecorder::posttrack
    2016-11-22 21:15:03.881 INFO  [1732801] [CRecorder::dump_brief@1246] CRecorder::dump_brief m_record_id       97 m_badflag     0 --dindex 
    2016-11-22 21:15:03.881 INFO  [1732801] [CRecorder::dump_brief@1254]  seqhis  8cccccccc9ccccd    TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA    
    2016-11-22 21:15:03.881 INFO  [1732801] [CRecorder::dump_brief@1264]  seqmat  343231323343231    Gd Ac LS Ac MO Ac Ac LS Ac Gd Ac LS Ac MO Ac - 
    2016-11-22 21:15:03.881 INFO  [1732801] [CRecorder::dump_brief@1259]  mskhis             1980    SA|DR|BT|TO

    ## DR in question is POST_SKIP MAT_SWAP  ... so issue is : how to handle StepTooSmall in Canned running in a way that matches Opticks ???
    ## hmm having a _SKIP means the MAT_SWAP is a mute point anyhow ???

    tp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.549 nm 430.000 mm/ns 194.519
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.638 nm 430.000 mm/ns 192.780
     )
    ( 3)  BT/BT     FrT                                                     
    [   3](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.638 nm 430.000 mm/ns 192.780
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.690 nm 430.000 mm/ns 197.134
     )
    ( 4)  BT/DR     LaR                                                     
    [   4](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.690 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )
    ( 5)  DR/NA     STS                                  POST_SKIP MAT_SWAP 
    [   5](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )
    ( 6)  NA/BT     FrT                                                     
    [   6](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3958.848  -513.480   322.311]  dir[   -0.871  -0.416   0.261]  pol[   -0.416   0.342  -0.842]  ns 31.802 nm 430.000 mm/ns 192.780
     )
    ( 7)  BT/BT     FrT                                                     
    [   7](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3958.848  -513.480   322.311]  dir[   -0.871  -0.416   0.261]  pol[   -0.416   0.342  -0.842]  ns 31.802 nm 430.000 mm/ns 192.780
     post               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3947.767  -518.773   325.634]  dir[   -0.869  -0.419   0.263]  pol[   -0.419   0.341  -0.842]  ns 31.868 nm 430.000 mm/ns 194.519
     )


     ## Live "pre" recording : 



Flags : NA NAN_ABORT actually means boundary status StepTooSmall
------------------------------------------------------------------

::

    158 #ifdef USE_CUSTOM_BOUNDARY
    159 unsigned int OpBoundaryFlag(const DsG4OpBoundaryProcessStatus status)
    160 #else
    161 unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status)
    162 #endif
    163 {
    164     unsigned flag = 0 ;
    165     switch(status)
    166     {
    167         case FresnelRefraction:
    168         case SameMaterial:
    169                                flag=BOUNDARY_TRANSMIT;
    170                                break;
    171         case TotalInternalReflection:
    172         case       FresnelReflection:
    173                                flag=BOUNDARY_REFLECT;
    174                                break;
    175         case StepTooSmall:
    176                                flag=NAN_ABORT;
    177                                break;
    178         case Absorption:
    179                                flag=SURFACE_ABSORB ;
    180                                break;
    181         case Detection:
    182                                flag=SURFACE_DETECT ;
    183                                break;
    184         case SpikeReflection:
    185                                flag=SURFACE_SREFLECT ;
    186                                break;
    187         case LobeReflection:
    188         case LambertianReflection:
    189                                flag=SURFACE_DREFLECT ;
    190                                break;
    191         case Undefined:
    192         case BackScattering:
    193         case NotAtBoundary:
    194         case NoRINDEX:
    195 




Differences between live and canned modes of CRecorder
---------------------------------------------------------

LiveRecording
      writes *pre*, until last step when writes both *post* also
      NB preFlag uses m_prior_boundary_status
      skips (pre) when prior boundary status is StepToSmall --> could just skip NA ?

CannedRecording
      writes *post*, except for first step when writes *pre* also 
      NB postFlag uses boundary_status, preFlag uses prior_boundary_status


Canned post writing Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    ( 3)  BT/*BT*     FrT                                                     
    [   3](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.638 nm 430.000 mm/ns 192.780
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.690 nm 430.000 mm/ns 197.134
     )
    ( 4)  BT/*DR*     LaR                                                     
    [   4](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.690 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )
           // step after DR will be skipped as STS:StepTooSmall SO NEED TO MAP_SWAP when next_boundary_status is STS 

    ( 5)  DR/*NA*     STS                                  POST_SKIP MAT_SWAP 
    [   5](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )
          // this step is (post)skipped as boundary_status is StepTooSmall

    ( 6)  NA/*BT*     FrT                                                     
    [   6](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3958.848  -513.480   322.311]  dir[   -0.871  -0.416   0.261]  pol[   -0.416   0.342  -0.842]  ns 31.802 nm 430.000 mm/ns 192.780
     )
    ( 7)  BT/*BT*     FrT                                                     
    [   7](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3958.848  -513.480   322.311]  dir[   -0.871  -0.416   0.261]  pol[   -0.416   0.342  -0.842]  ns 31.802 nm 430.000 mm/ns 192.780
     post               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3947.767  -518.773   325.634]  dir[   -0.869  -0.419   0.263]  pol[   -0.419   0.341  -0.842]  ns 31.868 nm 430.000 mm/ns 194.519
     )


Live pre writing 
~~~~~~~~~~~~~~~~~~~

::

    ( 3)  *BT*/BT     FrT                                                     
    [   3](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.638 nm 430.000 mm/ns 192.780
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.690 nm 430.000 mm/ns 197.134
     )
    ( 4)  *BT*/DR     LaR                                                     
    [   4](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.690 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )

    ( 5)  *DR*/NA     STS                                  POST_SKIP MAT_SWAP 
    [   5](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     )
        // material is swapped as boundary_status is StepTooSmall

    ( 6)  *NA*/BT     FrT                                                     
    [   6](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[   -0.863  -0.428   0.268]  pol[    0.428  -0.902  -0.062]  ns 25.712 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3958.848  -513.480   322.311]  dir[   -0.871  -0.416   0.261]  pol[   -0.416   0.342  -0.842]  ns 31.802 nm 430.000 mm/ns 192.780
     )
        // this step is (pre)skipped as prior_boundary_status is StepTooSmall


    ( 7)  *BT*/BT     FrT                                                     
    [   7](Stp ;opticalphoton stepNum   15(tk ;opticalphoton tid 98 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -2552.003-3636.7742282.802]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3958.848  -513.480   322.311]  dir[   -0.871  -0.416   0.261]  pol[   -0.416   0.342  -0.842]  ns 31.802 nm 430.000 mm/ns 192.780
     post               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3947.767  -518.773   325.634]  dir[   -0.869  -0.419   0.263]  pol[   -0.419   0.341  -0.842]  ns 31.868 nm 430.000 mm/ns 194.519
     )




CRecorder::LiveRecordStep::

     583     // shunt flags by 1 relative to steps, in order to set the generation code on first step
     584     // this doesnt miss flags, as record both pre and post at last step    
     585 
     586     unsigned preFlag = m_slot == 0 && m_stage == CStage::START ?
     587                                       m_gen
     588                                    :
     589                                       OpPointFlag(pre,  m_prior_boundary_status, m_stage )
     590                                    ;
     591 
     592     unsigned postFlag =               OpPointFlag(post, m_boundary_status      , m_stage );
     593 
     594 
     595     bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;
     596 
     597     bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;
     598 
     599     bool preSkip = m_prior_boundary_status == StepTooSmall && m_stage != CStage::REJOIN  ;
     600 
     601     bool matSwap = m_boundary_status == StepTooSmall ;
     602 
     603     unsigned preMat  = matSwap ? m_postmat : m_premat ;
     604 
     605     unsigned postMat = ( matSwap || m_postmat == 0 )  ? m_premat  : m_postmat ;
     606 
     607     if(surfaceAbsorb) postMat = m_postmat ;
     608 
     609     bool done = false ;
     610 
     611     // usually skip the pre, but the post becomes the pre at next step where will be taken 
     612     // 1-based material indices, so zero can represent None
     613     //
     614     //   RecordStepPoint records into m_slot (if < m_steps_per_photon) and increments m_slot
     615     // 
     616 
     617     if(lastPost)      m_step_action |= LAST_POST ;
     618     if(surfaceAbsorb) m_step_action |= SURF_ABS ;
     619     if(preSkip)       m_step_action |= PRE_SKIP ;
     620     if(matSwap)       m_step_action |= MAT_SWAP ;
     621 
     622 
     623     if(!preSkip)
     624     {
     625         m_step_action |= PRE_SAVE ;
     626         done = RecordStepPoint( pre, preFlag, preMat, m_prior_boundary_status, PRE );    // truncate OR absorb
     627         if(done) m_step_action |= PRE_DONE ;
     628     }
     629 
     630     if(lastPost && !done )
     631     {
     632         m_step_action |= POST_SAVE ;
     633         done = RecordStepPoint( post, postFlag, postMat, m_boundary_status, POST );
     634         if(done) m_step_action |= POST_DONE ;
     635     }



Canned
~~~~~~~~

CRecorder::CannedWriteSteps::

     728         CStage::CStage_t postStage = stage == CStage::REJOIN ? CStage::RECOLL : stage  ; // avoid duping the RE 
     729         postFlag = OpPointFlag(post, boundary_status, postStage);
     730 
     731         bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;
     732         bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;
     733 
     734         //bool postSkip = boundary_status == StepTooSmall && stage != CStage::REJOIN  ;  
     735         bool postSkip = boundary_status == StepTooSmall && !lastPost  ;
     736         bool matSwap = boundary_status == StepTooSmall ;
     737 
     738 
     739         if(lastPost)      m_step_action |= LAST_POST ;
     740         if(surfaceAbsorb) m_step_action |= SURF_ABS ;
     741         if(postSkip)      m_step_action |= POST_SKIP ;
     742         if(matSwap)       m_step_action |= MAT_SWAP ;
     743 
     744         switch(stage)
     745         {
     746             case CStage::START:  m_step_action |= STEP_START    ; break ;
     747             case CStage::REJOIN: m_step_action |= STEP_REJOIN   ; break ;
     748             case CStage::RECOLL: m_step_action |= STEP_RECOLL   ; break ;
     749             case CStage::COLLECT:                               ; break ;
     750             case CStage::UNKNOWN:assert(0)                      ; break ;
     751         }
     752 
     753 
     754         unsigned u_premat  = matSwap ? postmat : premat ;
     755         unsigned u_postmat = ( matSwap || postmat == 0 )  ? premat  : postmat ;
     756 
     757         if(surfaceAbsorb) u_postmat = postmat ;
     758 
     759         bool first = m_slot == 0 && stage == CStage::START ;
     760 
     761         if(stage == CStage::REJOIN)
     762         {
     763              decrementSlot();   // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
     764         }
     765 
     766        // as clearStp for each track, REJOIN will always be i=0
     767 
     768         preFlag = first ? m_gen : OpPointFlag(pre,  prior_boundary_status, stage) ;
     769 
     770         if(i == 0)
     771         {
     772             done = RecordStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE );
     773             done = RecordStepPoint( post, postFlag, u_postmat, boundary_status,       POST );
     774         }
     775         else
     776         {
     777             if(!postSkip)
     778             {
     779                 done = RecordStepPoint( post, postFlag, u_postmat, boundary_status, POST );
     780             }
     781         }



Trying to get canned closer to live recording with::

    -        bool postSkip = boundary_status == StepTooSmall && !lastPost  ;  
    +        bool postSkip = prior_boundary_status == StepTooSmall && !lastPost  ;  
     

Results in seqhis zeros with NA instead of BT following a DR::

    AB(1,torch,concentric)  None 0 
    A concentric/torch/  1 :  20161123-1103 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    B concentric/torch/ -1 :  20161123-1103 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
    .                seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000     61934.59/388 = 159.63  (pval:0.000 prob:1.000)  
    ...

       8             8e9ccccd             0         8679          8679.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT BT BT DR NA SA
       9              89ccccd          7540            0          7540.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT BT BT DR SA


      12      8cccccccc9ccccd          5339            0          5339.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
      13      8ccccccce9ccccd             0         5269          5269.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR NA BT BT BT BT BT BT BT SA




The kernel of the issue is that boundary status if a property between points, so moving from pre(live) to post(canned) recording 
requires translating "pre" style::

     598 
     599     bool preSkip = m_prior_boundary_status == StepTooSmall && m_stage != CStage::REJOIN  ;
     600 
     601     bool matSwap = m_boundary_status == StepTooSmall ;
     602 

Into "post" style::

     712     bool postSkip = boundary_status == StepTooSmall && !lastPost  ;
     713         
     714     bool matSwap = prior_boundary_status == StepTooSmall ;
     715         




