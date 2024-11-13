G4CXTest_GEOM_deviant_simple_history_TO_BT_BT_SA_at_1000_per_1M_level
========================================================================

Thoughts
---------

* this is using G4CXTest_GEOM.sh  starting from GDML, so could be chasing rabbits : confirm insitu before too much digging. 
* likely some coincident surface geometry behaviour difference


Investigation IDEAs in priority order
--------------------------------------

* WIP : revive simtrace cxt_min.sh to slice thru the sticks : to see the issue clearly

* TODO : make plot showing both input_photon hits plus the simtrace of the same stick 

* some boundaries look wrong : isnt the Steel within Acrylic not water ? 

  * implementing B side boundaries would be good 

* revive Geant4 "simtrace" equivalent for the sticks 

* use inputphotons from CD center so the same A,B slots have photons in the 
  same direction. Due to the simple history "TO BT BT SA" a large fraction of photons 
  will have "accidental" random alignment

* use inputphotons to target some sticks, for higher stats of the issue and
  less of the photons that behave as expected 

* get some event data into the raytrace vizualization view : so can see whats going down 


simtrace cxt_min.sh : cross section thru Fastener geometry
-----------------------------------------------------------

laptop::

    MODE=2 ~/o/cxt_min.sh ana



Issue : deviant simple history "TO BT BT SA"  : Opticks has 1000 more (out of 1M photons) with this history  
---------------------------------------------------------------------------------------------------------------

::

    ~/o/G4CXTest_GEOM.sh
    ~/o/G4CXTest_GEOM.sh chi2
    ...
    .                  BASH_SOURCE : /data/blyth/junotop/opticks/g4cx/tests/../../sysrap/tests/sseq_index_test.sh 
                              SDIR : /data/blyth/junotop/opticks/sysrap/tests 
                               TMP : /data/blyth/opticks 
                        EXECUTABLE : G4CXTest 
                           VERSION : 98 
                              BASE : /data/blyth/opticks/GEOM/J_2024aug27 
                              GEOM : J_2024aug27 
                            LOGDIR : /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98 
                             AFOLD : /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/A000 
                             BFOLD : /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/B000 
                              FOLD : /data/blyth/opticks/sseq_index_test 
    a_path $AFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/A000/seq.npy a_seq (1000000, 2, 2, )
    b_path $BFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/B000/seq.npy b_seq (1000000, 2, 2, )
    AB
    [sseq_index_ab::desc u.size 152063 opt BRIEF mode 6sseq_index_ab_chi2::desc sum  2301.5411 ndf 1801.0000 sum/ndf     1.2779 sseq_index_ab_chi2_ABSUM_MIN:40.0000
        TO AB                                                                                            :  126549 126392 :     0.0974 : Y :       2      4 :   
        TO BT BT BT BT BT BT SD                                                                          :   70475  70600 :     0.1108 : Y :      18     11 :   
        TO BT BT BT BT BT BT SA                                                                          :   57091  57086 :     0.0002 : Y :       5      1 :   
        TO SC AB                                                                                         :   51434  51597 :     0.2579 : Y :       4     30 :   
        TO SC BT BT BT BT BT BT SD                                                                       :   35876  36311 :     2.6213 : Y :      58     94 :   
        TO SC BT BT BT BT BT BT SA                                                                       :   29663  29733 :     0.0825 : Y :     124     53 :   
        TO SC SC AB                                                                                      :   19993  19819 :     0.7605 : Y :     137     51 :   

    :r:`TO BT BT SA                                                                                      :   19822  18585 :    39.8409 : Y :      71     72 : DEVIANT  `


        TO RE AB                                                                                         :   18319  18198 :     0.4009 : Y :       9      5 :   
        TO SC SC BT BT BT BT BT BT SD                                                                    :   15451  15529 :     0.1964 : Y :      19     22 :   
        TO SC SC BT BT BT BT BT BT SA                                                                    :   12785  12850 :     0.1648 : Y :      24    173 :   
        TO BT BT AB                                                                                      :   10955  10998 :     0.0842 : Y :      72     41 :   
        TO BT AB                                                                                         :    9253   9466 :     2.4237 : Y :      36     15 :   
        TO SC SC SC AB                                                                                   :    7544   7392 :     1.5469 : Y :      90      8 :   
        TO BT BT BT BT BT BT BT SA                                                                       :    7436   7473 :     0.0918 : Y :     176    144 :   
        TO RE BT BT BT BT BT BT SD                                                                       :    7417   7352 :     0.2861 : Y :     197     99 :   
        TO SC RE AB                                                                                      :    7137   7129 :     0.0045 : Y :     110     60 :   
        TO RE BT BT BT BT BT BT SA                                                                       :    7124   7049 :     0.3969 : Y :      48     35 :   
    :r:`TO SC BT BT SA                                                                                   :    6786   6159 :    30.3692 : Y :     120    126 : DEVIANT  `
        TO SC BT BT AB                                                                                   :    6375   6580 :     3.2439 : Y :     153     74 :   
        TO BT BT BT BT BT BT BT SR SA                                                                    :    6375   6315 :     0.2837 : Y :      16    184 :   
        TO SC SC SC BT BT BT BT BT BT SD                                                                 :    6146   6149 :     0.0007 : Y :     145      0 :   





laptop pyvista plotting
-------------------------

Plotting that history : clumps onto sticks apparent::

   PICK=AB HSEL="TO BT BT SA" SEL=0 ~/o/G4CXTest_GEOM.sh dna  


Viz check for targetting uni1
----------------------------------

Use viz to work out input photon targetting:: 

    MOI=uni1 EYE=0.1,0,5 ~/o/cx.sh

* for uni1 frame 0,0,5 is within LS directed up towards Acrylic and the underside of the stick foot.  
* hmm pick frame without the inversion ? 


A,B record step point check
-----------------------------

::

    wa = a.q_startswith("TO BT BT SA")
    wb = b.q_startswith("TO BT BT SA")
    ra = a.f.record[wa]
    rb = b.f.record[wb]

    In [25]: ra.shape
    Out[25]: (19822, 32, 4, 4)

    In [26]: rb.shape
    Out[26]: (18585, 32, 4, 4)
        

    In [42]: ra[0,:5,3].view(np.int32)
    Out[42]: 
    array([[       4096,           0,          71,        4096],
           [    6621184,           0,          71,        6144],
           [    6555648,           0,          71,        6144],
           [    7012480,           0, -2147483577,        6272],
           [          0,           0,           0,           0]], dtype=int32)

    In [46]: ra[0,:5,3].view(np.uint32) & 0x7fffffff
    Out[46]: 
    array([[   4096,       0,      71,    4096],
           [6621184,       0,      71,    6144],
           [6555648,       0,      71,    6144],
           [7012480,       0,      71,    6272],
           [      0,       0,       0,       0]], dtype=uint32)




    In [43]: rb[0,:5,3].view(np.int32)
    Out[43]: 
    array([[4096,    0,   72, 4096],
           [2048,    0,   72, 6144],
           [2048,    0,   72, 6144],
           [ 128,    0,   72, 6272],
           [   0,    0,    0,    0]], dtype=int32)





sphoton.h::

    +----+----------------+----------------+----------------+----------------+--------------------------+
    | q  |      x         |      y         |     z          |      w         |  notes                   |
    +====+================+================+================+================+==========================+
    |    |  pos.x         |  pos.y         |  pos.z         |  time          |                          |
    | q0 |                |                |                |                |                          |
    |    |                |                |                |                |                          |
    +----+----------------+----------------+----------------+----------------+--------------------------+
    |    |  mom.x         |  mom.y         | mom.z          |  iindex        |                          |
    | q1 |                |                |                | (unsigned)     |                          |
    |    |                |                |                |                |                          |
    +----+----------------+----------------+----------------+----------------+--------------------------+
    |    |  pol.x         |  pol.y         |  pol.z         |  wavelength    |                          |
    | q2 |                |                |                |                |                          |
    |    |                |                |                |                |                          |
    +----+----------------+----------------+----------------+----------------+--------------------------+
    |    | boundary_flag  |  identity      |  orient_idx    |  flagmask      |  (unsigned)              |
    | q3 | (3,0)          |                |  orient:1bit   |                |                          |
    |    |                |                |                |                |                          |
    +----+----------------+----------------+----------------+----------------+--------------------------+






Check the boundaries
---------------------

* note that B lacks the boundary info

::

    P[blyth@localhost opticks]$ ~/o/bin/bd_names.sh
    /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/SSim/stree/standard
    0    Galactic///Galactic
    1    Galactic///Rock
    2    Rock///Galactic
    3    Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air
    4    Rock///Rock
    ..
    96   vetoWater/Implicit_RINDEX_NoRINDEX_pWaterPool_ZC2.A03_A03_HBeam_phys//LatticedShellSteel
    97   vetoWater/Implicit_RINDEX_NoRINDEX_pWaterPool_ZC2.A05_A05_HBeam_phys//LatticedShellSteel
    98   Air/CDTyvekSurface//Tyvek
    99   Tyvek//CDInnerTyvekSurface/Water
    100  Water///Acrylic
    101  Acrylic///LS
    102  LS///Acrylic
    103  LS///PE_PA
    104  Water/StrutAcrylicOpSurface//StrutSteel
    105  Water/Strut2AcrylicOpSurface//StrutSteel
    106  Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lSteel_phys//Steel
    107  Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lFasteners_phys//Steel
    108  Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lUpper_phys//Steel
    109  Water///PE_PA
    110  Water///Water



    99   Tyvek//CDInnerTyvekSurface/Water
    101  Acrylic///LS
    100  Water///Acrylic

    107  Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lFasteners_phys//Steel
    108  Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lUpper_phys//Steel


    In [23]: np.c_[np.unique( ra[:,3,3,0].view(np.uint32) >> 16, return_counts=True )]
    Out[23]: 
    array([[   99, 14137],            ## Tyvek//CDInnerTyvekSurface/Water
           [  107,  3828],            ## Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lFasteners_phys//Steel
           [  108,  1857]])           ## Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lUpper_phys//Steel


* HUH: isnt the Steel within Acrylic not water ? 


HMM, having boundary for B would be handy::


          Tyvek 
          -----------3:SA----------------------------------   19629   (+1805)    



                                                    
                                 +-------------+              17964    (+127)    
                                /   Steel       \
                               +-----------------+            17837    ( +13)
          Water 
          -----------2:BT----------------------------------   17824   ( +124) 
          Acrylic 
          ---------- 1:BT----------------------------------   17700
          LS

                     0:TO                    


Using 2D viz simtrace for uni1:0:0 shows those radial offsets to correspond to the IonRing::

   MODE=2 ~/o/cxt_min.sh ana 

::

    P[blyth@localhost tests]$ PICK=AB HSEL="TO BT BT SA" ~/o/G4CXTest_GEOM.sh ana


    In [15]: ra[:100,:4,3,0].view(np.uint32) >> 16
    Out[15]: 
    array([[  0, 101, 100, 107],
           [  0, 101, 100, 107],
           [  0, 101, 100,  99],
           [  0, 101, 100,  99],
           [  0, 101, 100,  99],
           [  0, 101, 100,  99],
           [  0, 101, 100, 107],
           [  0, 101, 100,  99],
           [  0, 101, 100, 107],
           [  0, 101, 100,  99],
           [  0, 101, 100,  99],
           [  0, 101, 100,  99],
           [  0, 101, 100, 108],
           [  0, 101, 100,  99],

::

    In [20]: np.c_[np.unique( ra[:,0,3,0].view(np.uint32) >> 16, return_counts=True )]
    Out[20]: array([[    0, 19822]])

    In [21]: np.c_[np.unique( ra[:,1,3,0].view(np.uint32) >> 16, return_counts=True )]
    Out[21]: array([[  101, 19822]])

    In [22]: np.c_[np.unique( ra[:,2,3,0].view(np.uint32) >> 16, return_counts=True )]
    Out[22]: array([[  100, 19822]])

    In [23]: np.c_[np.unique( ra[:,3,3,0].view(np.uint32) >> 16, return_counts=True )]
    Out[23]: 
    array([[   99, 14137],
           [  107,  3828],
           [  108,  1857]])



Check the radii, Tyvek ones should be larger::

    In [43]: np.sqrt(np.sum(ra[:,:4,0,:3]*ra[:,:4,0,:3],axis=2))
    Out[43]: 
    array([[  100.   , 17700.002, 17824.   , 17838.041],
           [  100.   , 17700.   , 17823.998, 17837.855],
           [  100.   , 17699.996, 17824.   , 19629.   ],
           [  100.   , 17700.   , 17824.   , 19629.   ],
           [  100.   , 17700.   , 17824.   , 19629.   ],
           ...,
           [  100.   , 17700.002, 17824.   , 19629.   ],
           [  100.   , 17700.   , 17824.   , 19629.   ],
           [  100.   , 17700.   , 17824.   , 19629.   ],
           [  100.   , 17700.002, 17824.   , 19629.   ],
           [  100.   , 17699.998, 17824.   , 19628.998]], dtype=float32)



Tight groupings for first 3::

    In [15]: np.unique(rra[:,0], return_counts=True)
    Out[15]: 
    (array([100., 100., 100., 100., 100., 100., 100.], dtype=float32),
     array([   4,  544, 1395, 9848, 6810, 1213,    8]))

    In [16]: np.unique(rra[:,1], return_counts=True)
    Out[16]: 
    (array([17699.994, 17699.996, 17699.998, 17700.   , 17700.002, 17700.004], dtype=float32),
     array([    1,    40,  1536, 11342,  6880,    23]))

    In [17]: np.unique(rra[:,2], return_counts=True)
    Out[17]: 
    (array([17823.996, 17823.998, 17824.   , 17824.002], dtype=float32),
     array([    5,   806, 18879,   132]))



    In [20]: np.c_[np.unique(rra[:,3].astype(np.int32), return_counts=True)]
    Out[20]: 
    array([
           [17837,  2835],
           [17838,   991],
           [17839,     1],      ## A has lots more at low radii  
           [17851,     1],      ## looks like mostly boundry 107 

           [17964,  1857],

           [19628,  4286],
           [19629,  9851]])


    ## low radii mostly boundary 107 ?

    In [30]: np.c_[np.unique(rra[:,3][ba[:,3] == 107].astype(np.int32), return_counts=True)]
    Out[30]: 
    array([[17837,  2835],
           [17838,   991],
           [17839,     1],
           [17851,     1]])


    ## mid radii mostly boundary 108 

    In [32]: np.c_[np.unique(rra[:,3][ba[:,3] == 108].astype(np.int32), return_counts=True)]
    Out[32]: array([[17964,  1857]])


    ## high radii mostly boundary 99 Tyvek 

    In [31]: np.c_[np.unique(rra[:,3][ba[:,3] == 99].astype(np.int32), return_counts=True)]
    Out[31]: 
    array([[19628,  4286],
           [19629,  9851]])




    In [21]: np.c_[np.unique(rrb[:,3].astype(np.int32), return_counts=True)]
    Out[21]: 
    array([[17824,     2],
           [17825,     2],
           [17826,     2],

           [17847,     1],       ##  B has a smattering at low radii
           [17848,     1],
           [17849,     1],
           [17853,     1],
           [17893,     1],


           [17964,  4452],
           [17965,   254],

           [19628,  1869],
           [19629, 11997],

           [22253,     2]])


    ## B has very few at low radii, more at mid and high 
    ## A has many at low radii  


    In [23]: np.c_[np.unique( ra[:,3,3,0].view(np.uint32) >> 16, return_counts=True )]
    Out[23]: 
    array([[   99, 14137],            ## Tyvek//CDInnerTyvekSurface/Water
           [  107,  3828],            ## Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lFasteners_phys//Steel
           [  108,  1857]])           ## Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lUpper_phys//Steel




Expected the Steel to be within Acrylic not Water
---------------------------------------------------

Look into this over in ~/j/setupCD_Sticks_Fastener/Fastener_asis_sibling_soup.rst


