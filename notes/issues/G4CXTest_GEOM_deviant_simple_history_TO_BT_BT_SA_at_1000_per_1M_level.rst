G4CXTest_GEOM_deviant_simple_history_TO_BT_BT_SA_at_1000_per_1M_level
========================================================================

Thoughts
---------

* this is using G4CXTest_GEOM.sh  starting from GDML, so could be chasing rabbits : confirm insitu before too much digging. 
* likely some coincident surface geometry behaviour difference


Investigation IDEAs in priority order
--------------------------------------

* DONE : revive simtrace cxt_min.sh to slice thru the sticks : to see the issue clearly

* TODO : inputphotons to targetting some sticks, focussing on the issue 

* TODO : make plot showing both input_photon hits plus the simtrace of the same stick 

* DONE : ADDED CONFIG TO FIX : some boundaries look wrong : uni1 Steel should be within Acrylic not water 

  * impl geometry with fixed uni1 heirarchy option to compare current with 
  * ~/j/setupCD_Sticks_Fastener/Fastener_asis_sibling_soup.rst

* revive Geant4 "simtrace" equivalent for the sticks 

* use inputphotons from CD center so the same A,B slots have photons in the 
  same direction. Due to the simple history "TO BT BT SA" a large fraction of photons 
  will have "accidental" random alignment

* impl compositing of event data onto raytrace vizualization view : so can vizualize such issues directly 

* MAYBE : implement B side boundary recording in U4Recorder



DONE : simtrace cxt_min.sh : cross section thru Fastener geometry
----------------------------------------------------------------------

workstation::

    ~/o/cxt_min.sh
    MODE=2 ~/o/cxt_min.sh ana  ## matplotlib 2D
    ##pyvista not installed on workstation

laptop::

    ~/o/cxt_min.sh grab

    MODE=3 ~/o/cxt_min.sh ana  ## pyvista 3D
    MODE=2 ~/o/cxt_min.sh ana  ## matplotlib 2D



Issue : deviant simple history "TO BT BT SA"  : Opticks has 1000 more (out of 1M photons) with this history  
---------------------------------------------------------------------------------------------------------------

::

    ~/o/G4CXTest_GEOM.sh             ## A-B comparison starting from GDML
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
        ##   A-B  +1237

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






fix still working with the water layer
----------------------------------------

With jok.bash geometry config::

    150    : jcv FastenerConstruction
    151    unset FastenerConstruction__CONFIG
    152 
    153    local FC_ASIS=0                       ## geometry present but does not render
    154    local FC_MULTIUNION_CONTIGUOUS=1
    155    local FC_MULTIUNION_DISCONTIGUOUS=2   ## G4MultiUnion SEGV with input photons
    156    local FC_LISTNODE_DISCONTIGUOUS=3     ## avoid the G4MultiUnion but still translate to listnode
    157    local FC_LISTNODE_CONTIGUOUS=4
    158 
    159    #export FastenerConstruction__CONFIG=$FC_ASIS
    160    #export FastenerConstruction__CONFIG=$FC_MULTIUNION_DISCONTIGUOUS  
    161    export FastenerConstruction__CONFIG=$FC_LISTNODE_DISCONTIGUOUS
    162 
    163 
    164 
    165    unset LSExpDetectorConstruction__setupCD_Sticks_Fastener_CONFIG
    166    local AAF_ASIS=0
    167    local AAF_HIERARCHY=1
    168    export LSExpDetectorConstruction__setupCD_Sticks_Fastener_CONFIG=$AAF_HIERARCHY
    169 
    170 
    171    unset LSExpDetectorConstruction__setupCD_Sticks_Fastener_Hierarchy_DELTA_MM 
    172    #local FC_DELTA_MM_DEFAULT=0.10
    173    #local FC_DELTA_MM_ENLARGED_FOR_VISIBILITY=2
    174    #export LSExpDetectorConstruction__setupCD_Sticks_Fastener_Hierarchy_DELTA_MM=$FC_DELTA_MM_ENLARGED_FOR_VISIBILITY
    175 


Persist geometry including GDML with jok run::

    jok-;jok-tds-gdb 

The do the from GDML run and A-B chi2 comparison::

    ~/o/G4CXTest_GEOM.sh
    ~/o/G4CXTest_GEOM.sh chi2

::

    P[blyth@localhost sysrap]$ ~/o/G4CXTest_GEOM.sh chi2
    knobs is a function
    knobs () 
    { 
        type $FUNCNAME;
        local exceptionFlags;
        local debugLevel;
        local optLevel;
        exceptionFlags=NONE;
        debugLevel=NONE;
        optLevel=LEVEL_3;
        export PIP__CreatePipelineOptions_exceptionFlags=$exceptionFlags;
        export PIP__CreateModule_debugLevel=$debugLevel;
        export PIP__linkPipeline_debugLevel=$debugLevel;
        export PIP__CreateModule_optLevel=$optLevel
    }
                       BASH_SOURCE : /data/blyth/junotop/opticks/g4cx/tests/../../sysrap/tests/sseq_index_test.sh 
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
    [sseq_index_ab::desc u.size 152006 opt BRIEF mode 6sseq_index_ab_chi2::desc sum  1970.0301 ndf 1823.0000 sum/ndf     1.0807 sseq_index_ab_chi2_ABSUM_MIN:40.0000
        TO AB                                                                                            :  126549 127238 :     1.8705 : Y :       2      0 :   
        TO BT BT BT BT BT BT SD                                                                          :   70475  70420 :     0.0215 : Y :      18      1 :   
        TO BT BT BT BT BT BT SA                                                                          :   57092  56955 :     0.1646 : Y :       5      9 :   
        TO SC AB                                                                                         :   51434  51096 :     1.1142 : Y :       4     49 :   
        TO SC BT BT BT BT BT BT SD                                                                       :   35876  36125 :     0.8611 : Y :      58    104 :   
        TO SC BT BT BT BT BT BT SA                                                                       :   29662  29855 :     0.6259 : Y :     124     25 :   
        TO SC SC AB                                                                                      :   19993  19993 :     0.0000 : Y :     137     40 :   
        TO RE AB                                                                                         :   18319  18320 :     0.0000 : Y :       9     18 :   
        TO BT BT SA                                                                                      :   15985  15716 :     2.2826 : Y :     205     79 :   
        TO SC SC BT BT BT BT BT BT SD                                                                    :   15451  15354 :     0.3054 : Y :      19     43 :   
        TO SC SC BT BT BT BT BT BT SA                                                                    :   12785  12801 :     0.0100 : Y :      24     26 :   
        TO BT BT AB                                                                                      :   10967  10899 :     0.2115 : Y :      72     71 :   
        TO BT AB                                                                                         :    9253   9402 :     1.1901 : Y :      36     19 :   
        TO BT BT BT SA                                                                                   :    9104   9020 :     0.3893 : Y :      71    747 :   
        TO BT BT BT BT BT BT BT SA                                                                       :    7435   7642 :     2.8420 : Y :     176    265 :   
        TO SC SC SC AB                                                                                   :    7544   7413 :     1.1474 : Y :      90    307 :   
        TO RE BT BT BT BT BT BT SD                                                                       :    7417   7376 :     0.1136 : Y :     197     10 :   
        TO SC RE AB                                                                                      :    7137   7216 :     0.4348 : Y :     110    209 :   
        TO RE BT BT BT BT BT BT SA                                                                       :    7124   6974 :     1.5960 : Y :      48    220 :   
        TO SC BT BT AB                                                                                   :    6384   6494 :     0.9396 : Y :     153     33 :   
        TO BT BT BT BT BT BT BT SR SA                                                                    :    6375   6430 :     0.2362 : Y :      16     73 :   
        TO SC SC SC BT BT BT BT BT BT SD                                                                 :    6146   6302 :     1.9550 : Y :     145     17 :   
        TO BT BT BT BT SD                                                                                :    6147   5989 :     2.0570 : Y :      13    285 :   
        TO SC BT AB                                                                                      :    5595   5762 :     2.4557 : Y :       8    329 :   
        TO BT BT DR BT SA                                                                                :    5445   5558 :     1.1605 : Y :     600     78 :   
        TO RE RE AB                                                                                      :    5539   5390 :     2.0314 : Y :     267    214 :   
        TO SC SC SC BT BT BT BT BT BT SA                                                                 :    5084   5166 :     0.6560 : Y :      23    240 :   
        TO SC BT BT SA                                                                                   :    4803   4886 :     0.7110 : Y :     120     97 :   
        TO SC BT BT BT BT BT BT BT SA                                                                    :    4446   4425 :     0.0497 : Y :      20    256 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                               :    3805   3825 :     0.0524 : Y :     362    345 :   
        TO RE SC AB                                                                                      :    3660   3493 :     3.8989 : Y :      54     93 :   
        TO SC RE BT BT BT BT BT BT SD                                                                    :    3190   3200 :     0.0156 : Y :     292    110 :   
        TO SC BT BT BT BT BT BT BT SR SA                                                                 :    3153   3176 :     0.0836 : Y :     243    139 :   
        TO SC BT BT BT SA                                                                                :    3171   3134 :     0.2171 : Y :     121    135 :   
        TO BT BT BT BT BT BT BT SD                                                                       :    3129   3136 :     0.0078 : Y :     181     74 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                               :    3133   3082 :     0.4185 : Y :      22    531 :   
        TO BT BT BT BT BT BT BT SR SR SA                                                                 :    3049   3058 :     0.0133 : Y :     286     57 :   
        TO SC SC BT BT AB                                                                                :    2869   2930 :     0.6417 : Y :     636     90 :   
        TO BT BT BT BT AB                                                                                :    2913   2848 :     0.7334 : Y :     225    460 :   
        TO SC RE BT BT BT BT BT BT SA                                                                    :    2877   2900 :     0.0916 : Y :     151      3 :   
        TO SC BT BT BT BT SD                                                                             :    2843   2831 :     0.0254 : Y :     224    696 :   
        TO RE SC BT BT BT BT BT BT SD                                                                    :    2827   2813 :     0.0348 : Y :     282    481 :   
        TO SC SC SC SC AB                                                                                :    2745   2781 :     0.2345 : Y :     142    431 :   
        TO SC SC BT AB                                                                                   :    2712   2761 :     0.4387 : Y :     987    616 :   
        TO SC SC RE AB                                                                                   :    2626   2675 :     0.4529 : Y :     445     23 :   
        TO RE SC BT BT BT BT BT BT SA                                                                    :    2618   2562 :     0.6054 : Y :     781    601 :   
        TO SC SC SC SC BT BT BT BT BT BT SD                                                              :    2338   2354 :     0.0546 : Y :      59    101 :   
        TO RE RE BT BT BT BT BT BT SD                                                                    :    2229   2238 :     0.0181 : Y :     655    574 :   
        TO BT BT BT BT BT BT BT SR SR SR SA                                                              :    2208   2026 :     7.8233 : Y :     528    314 :   
        TO RE RE BT BT BT BT BT BT SA                                                                    :    2180   2132 :     0.5343 : Y :    1501     98 :   
        TO SC RE RE AB                                                                                   :    2118   2103 :     0.0533 : Y :    1340    286 :   
        TO SC BT BT BT BT BT BT BT SD                                                                    :    2048   2048 :     0.0000 : Y :     876    213 :   
        TO SC SC BT BT BT BT BT BT BT SA                                                                 :    2018   1989 :     0.2099 : Y :     851    911 :   
        TO SC BT BT BT BT SA                                                                             :    1978   1940 :     0.3686 : Y :     799   1082 :   
        TO SC SC BT BT SA                                                                                :    1915   1962 :     0.5698 : Y :     772    501 :   
        TO SC SC SC SC BT BT BT BT BT BT SA                                                              :    1943   1893 :     0.6517 : Y :     525     95 :   
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                            :    1802   1762 :     0.4489 : Y :     448    510 :   
        TO RE RE RE AB                                                                                   :    1610   1622 :     0.0446 : Y :    1230    406 :   
        TO RE BT BT AB                                                                                   :    1523   1484 :     0.5058 : Y :    1117    462 :   
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                            :    1482   1485 :     0.0030 : Y :    1225    762 :   
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 152006 opt AZERO mode 1
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                         :      -1     16 :     0.0000 : N :      -1  24615 : AZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                            :      -1     13 :     0.0000 : N :      -1  73425 : AZERO C2EXC  
        TO BT BT SR BT BT AB                                                                             :      -1     12 :     0.0000 : N :      -1  67480 : AZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SR BT SD                                         :      -1     11 :     0.0000 : N :      -1  71722 : AZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 152006 opt BZERO mode 2
        TO BT BT DR BT BT BT SD                                                                          :      26     -1 :     0.0000 : N :    1930     -1 : BZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                               :      15     -1 :     0.0000 : N :   12882     -1 : BZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                            :      13     -1 :     0.0000 : N :    5370     -1 : BZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BT SD                                         :      13     -1 :     0.0000 : N :   54864     -1 : BZERO C2EXC  
        TO SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SA                                      :      12     -1 :     0.0000 : N :   42959     -1 : BZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 152006 opt DEVIANT mode 5
    :r:`TO BT BR BT AB                                                                                   :     144     95 :    10.0460 : Y :    2809   2734 : DEVIANT  `
    :r:`TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT AB                                            :     115     69 :    11.5000 : Y :    5254   3849 : DEVIANT  `
    :r:`TO SC SC SC SC BT BT BT BT BT BT BT BT SD                                                        :      61    104 :    11.2061 : Y :    2931  10623 : DEVIANT  `
    :r:`TO BT BT BT BT BT BT BT SR BT BT BT BT BT BT BT BT BT BT BT BT BT SA                             :      38     74 :    11.5714 : Y :   21512  11070 : DEVIANT  `
    :r:`TO BT BT BT BT BT DR BT SA                                                                       :      65     33 :    10.4490 : Y :    4302  18353 : DEVIANT  `
    :r:`TO BT BT BT BT DR BT BT BT SA                                                                    :      23     54 :    12.4805 : Y :   92725  12466 : DEVIANT  `
    :r:`TO BT BT BT BT BT BT BT BT SD                                                                    :      47      1 :    44.0833 : Y :   11355 814025 : DEVIANT  `
    :r:`TO BT BT BT BT BR BR BR DR AB                                                                    :       5     36 :    23.4390 : Y :  185265  19130 : DEVIANT  `
    :r:`TO RE SC SC SC SC BT BT AB                                                                       :      32     11 :    10.2558 : Y :   16749  69793 : DEVIANT  `
    :r:`TO BT BT BT BT BT BT BT SR SR SR SR SR BT BT BT BT BT AB                                         :      32     11 :    10.2558 : Y :   34403  47012 : DEVIANT  `
    ]sseq_index_ab::desc

    f

    CMDLINE:/data/blyth/junotop/opticks/sysrap/tests/sseq_index_test.py
    f.base:/data/blyth/opticks/sseq_index_test

      : f.sseq_index_ab_chi2                               :                 (4,) : 0:00:01.034498 

     min_stamp : 2024-11-26 11:22:03.578964 
     max_stamp : 2024-11-26 11:22:03.578964 
     dif_stamp : 0:00:00 
     age_stamp : 0:00:01.034498 
    [1970.03 1823.     40.      0.  ]
    c2sum/c2n:c2per(C2CUT)  1970.03/1823:1.081 (40) pv[1.000,> 0.05 : null-hyp ] 
    c2sum :  1970.0301 c2n :  1823.0000 c2per:     1.0807  C2CUT:   40 
    P[blyth@localhost sysrap]$ 








hierarchy fix and without the AdditionAcrylicConstruction__rdelta_mm : slightly bigger chi2
-----------------------------------------------------------------------------------------------

:: 

    unset AdditionAcrylicConstruction__rdelta_mm  


    jok-;jok-tds-gdb              # FRESH TAB
    ~/o/cxt_min.sh                # FRESH TAB
    ~/o/G4CXTest_GEOM.sh          # FRESH TAB
    ~/o/G4CXTest_GEOM.sh chi2


    a_path $AFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/A000/seq.npy a_seq (1000000, 2, 2, )
    b_path $BFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/B000/seq.npy b_seq (1000000, 2, 2, )
    AB
    [sseq_index_ab::desc u.size 151013 opt BRIEF mode 6sseq_index_ab_chi2::desc sum  1934.8416 ndf 1819.0000 sum/ndf     1.0637 sseq_index_ab_chi2_ABSUM_MIN:40.0000
        TO AB                                                                                            :  126549 127117 :     1.2718 : Y :       2     12 :   
        TO BT BT BT BT BT BT SD                                                                          :   70475  70552 :     0.0420 : Y :      18     25 :   
        TO BT BT BT BT BT BT SA                                                                          :   57091  57381 :     0.7347 : Y :       5      1 :   
        TO SC AB                                                                                         :   51434  51490 :     0.0305 : Y :       4      2 :   
        TO SC BT BT BT BT BT BT SD                                                                       :   35876  35849 :     0.0102 : Y :      58     13 :   
        TO SC BT BT BT BT BT BT SA                                                                       :   29663  29775 :     0.2110 : Y :     124     10 :   
        TO SC SC AB                                                                                      :   19993  19826 :     0.7004 : Y :     137    247 :   
        TO BT BT SA                                                                                      :   19804  19339 :     5.5240 : Y :      71     78 :   
        TO RE AB                                                                                         :   18319  18376 :     0.0885 : Y :       9     70 :   
        TO SC SC BT BT BT BT BT BT SD                                                                    :   15451  15501 :     0.0808 : Y :      19     51 :   
        TO SC SC BT BT BT BT BT BT SA                                                                    :   12785  12995 :     1.7106 : Y :      24     54 :   
        TO BT BT AB                                                                                      :   10967  10978 :     0.0055 : Y :      72     73 :   
        TO BT AB                                                                                         :    9253   9245 :     0.0035 : Y :      36      4 :   
        TO SC SC SC AB                                                                                   :    7544   7592 :     0.1522 : Y :      90     45 :   
        TO BT BT BT BT BT BT BT SA                                                                       :    7436   7497 :     0.2492 : Y :     176     36 :   
        TO RE BT BT BT BT BT BT SD                                                                       :    7417   7337 :     0.4338 : Y :     197     84 :   
        TO SC RE AB                                                                                      :    7137   7239 :     0.7237 : Y :     110    140 :   
        TO RE BT BT BT BT BT BT SA                                                                       :    7124   7121 :     0.0006 : Y :      48    102 :   
        TO SC BT BT SA                                                                                   :    6772   6889 :     1.0020 : Y :     120    240 :   
        TO BT BT BT BT BT BT BT SR SA                                                                    :    6375   6414 :     0.1189 : Y :      16    219 :   
        TO SC BT BT AB                                                                                   :    6384   6233 :     1.8072 : Y :     153     34 :   
        TO BT BT BT BT SD                                                                                :    6147   6119 :     0.0639 : Y :      13      6 :   
        TO SC SC SC BT BT BT BT BT BT SD                                                                 :    6146   6141 :     0.0020 : Y :     145      0 :   
        TO SC BT AB                                                                                      :    5595   5832 :     4.9155 : Y :       8     37 :   
        TO BT BT DR BT SA                                                                                :    5456   5546 :     0.7362 : Y :     600    570 :   
        TO RE RE AB                                                                                      :    5539   5371 :     2.5870 : Y :     267    210 :   
        TO BT BT BT SA                                                                                   :    5303   5053 :     6.0351 : Y :     745      7 :   
        TO SC SC SC BT BT BT BT BT BT SA                                                                 :    5084   4944 :     1.9545 : Y :      23    117 :   
        TO SC BT BT BT BT BT BT BT SA                                                                    :    4446   4311 :     2.0812 : Y :      20     77 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                               :    3805   3734 :     0.6687 : Y :     362     18 :   
        TO RE SC AB                                                                                      :    3660   3450 :     6.2025 : Y :      54    143 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                               :    3157   3236 :     0.9762 : Y :      22    730 :   
        TO SC RE BT BT BT BT BT BT SD                                                                    :    3190   3107 :     1.0940 : Y :     292    162 :   
        TO SC BT BT BT BT BT BT BT SR SA                                                                 :    3153   3164 :     0.0192 : Y :     243    186 :   
        TO BT BT BT BT BT BT BT SR SR SA                                                                 :    3049   3140 :     1.3380 : Y :     286    847 :   
        TO BT BT BT BT BT BT BT SD                                                                       :    3129   3058 :     0.8148 : Y :     181   1054 :   
        TO SC RE BT BT BT BT BT BT SA                                                                    :    2877   3033 :     4.1178 : Y :     151    678 :   
        TO SC SC BT BT AB                                                                                :    2869   2988 :     2.4178 : Y :     636    180 :   
        TO SC BT BT BT BT SD                                                                             :    2843   2962 :     2.4394 : Y :     224    399 :   
        TO BT BT BT BT AB                                                                                :    2913   2822 :     1.4439 : Y :     225     59 :   
        TO RE SC BT BT BT BT BT BT SD                                                                    :    2827   2870 :     0.3246 : Y :     282    327 :   
        TO SC SC BT BT SA                                                                                :    2782   2782 :     0.0000 : Y :     772     19 :   
        TO SC SC RE AB                                                                                   :    2626   2757 :     3.1880 : Y :     445    268 :   
        TO SC SC SC SC AB                                                                                :    2745   2757 :     0.0262 : Y :     142   1011 :   
        TO SC SC BT AB                                                                                   :    2712   2744 :     0.1877 : Y :     987    752 :   
        TO RE SC BT BT BT BT BT BT SA                                                                    :    2619   2603 :     0.0490 : Y :     781     30 :   
        TO SC SC SC SC BT BT BT BT BT BT SD                                                              :    2338   2358 :     0.0852 : Y :      59    621 :   
        TO RE RE BT BT BT BT BT BT SD                                                                    :    2229   2238 :     0.0181 : Y :     655    283 :   
        TO BT BT BT BT BT BT BT SR SR SR SA                                                              :    2208   2213 :     0.0057 : Y :     528     32 :   
        TO RE RE BT BT BT BT BT BT SA                                                                    :    2180   2114 :     1.0144 : Y :    1501    823 :   
        TO SC RE RE AB                                                                                   :    2118   2117 :     0.0002 : Y :    1340     44 :   
        TO SC SC BT BT BT BT BT BT BT SA                                                                 :    2018   2059 :     0.4123 : Y :     851   1060 :   
        TO SC BT BT BT BT BT BT BT SD                                                                    :    2049   2055 :     0.0088 : Y :     876   1419 :   
        TO SC BT BT BT BT SA                                                                             :    1964   1973 :     0.0206 : Y :     799    591 :   
        TO SC SC SC SC BT BT BT BT BT BT SA                                                              :    1943   1893 :     0.6517 : Y :     525   1403 :   
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                            :    1802   1770 :     0.2867 : Y :     448     94 :   
        TO RE BT BT SA                                                                                   :    1730   1751 :     0.1267 : Y :     608     17 :   
        TO RE RE RE AB                                                                                   :    1610   1638 :     0.2414 : Y :    1230    148 :   
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                            :    1494   1526 :     0.3391 : Y :    1225   1894 :   
        TO RE BT BT AB                                                                                   :    1523   1484 :     0.5058 : Y :    1117    301 :   
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 151013 opt AZERO mode 1
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                            :      -1     23 :     0.0000 : N :      -1  30794 : AZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                         :      -1     19 :     0.0000 : N :      -1  64134 : AZERO C2EXC  
        TO BT BT BT BT BT BT BT SR SR SR BR BT BT BT BT BT BT BT BT BT BT BT SA                          :      -1     13 :     0.0000 : N :      -1  51825 : AZERO C2EXC  
        TO BT BT SR BT BT AB                                                                             :      -1     12 :     0.0000 : N :      -1   6286 : AZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 151013 opt BZERO mode 2
        TO BT BT DR BT BT BT SD                                                                          :      26     -1 :     0.0000 : N :    1930     -1 : BZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BT SD                                            :      21     -1 :     0.0000 : N :   10972     -1 : BZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                               :      15     -1 :     0.0000 : N :   12882     -1 : BZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                            :      13     -1 :     0.0000 : N :    5370     -1 : BZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 151013 opt DEVIANT mode 5
    :r:`TO RE BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                            :     358    450 :    10.4752 : Y :   10662   2511 : DEVIANT  `
    :r:`TO SC BT BT BT BT BR BT BT DR BT SA                                                              :     178    120 :    11.2886 : Y :    2976   5602 : DEVIANT  `
    :r:`TO SC SC SC RE RE BT BT BT BT BT BT SA                                                           :     101    152 :    10.2806 : Y :   10903   3288 : DEVIANT  `
    :r:`TO BT BT BT BT BT BT BT BT SD                                                                    :      47      3 :    38.7200 : Y :   11355  83098 : DEVIANT  `
    :r:`TO BT BT BT BT BR BR BR DR AB                                                                    :       5     37 :    24.3810 : Y :  185265  54753 : DEVIANT  `
    ]sseq_index_ab::desc

    f

    CMDLINE:/data/blyth/junotop/opticks/sysrap/tests/sseq_index_test.py
    f.base:/data/blyth/opticks/sseq_index_test

      : f.sseq_index_ab_chi2                               :                 (4,) : 0:00:00.947983 

     min_stamp : 2024-11-14 18:50:54.083360 
     max_stamp : 2024-11-14 18:50:54.083360 
     dif_stamp : 0:00:00 
     age_stamp : 0:00:00.947983 
    [1934.842 1819.      40.       0.   ]
    c2sum/c2n:c2per(C2CUT)  1934.84/1819:1.064 (40) pv[1.000,> 0.05 : null-hyp ] 
    c2sum :  1934.8416 c2n :  1819.0000 c2per:     1.0637  C2CUT:   40 



With hierarchy fix and adhoc flip AND AdditionAcrylicConstruction__rdelta_mm = 1 
--------------------------------------------------------------------------------------

::

    export AdditionAcrylicConstruction__rdelta_mm=1 

::

    P[blyth@localhost ~]$ jok-;jok-tds-gdb 

    ## CAUTION : DO THIS IN FRESH TAB
    P[blyth@localhost ~]$ ~/o/cxt_min.sh    ## simtrace for geometry slice check 

    ## CAUTION : AGAIN FRESH TAB : TO AVOID ENV INTERFERENCE
    P[blyth@localhost ~]$ ~/o/G4CXTest_GEOM.sh
    P[blyth@localhost ~]$ ~/o/G4CXTest_GEOM.sh chi2
    ...

    a_path $AFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/A000/seq.npy a_seq (1000000, 2, 2, )
    b_path $BFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/B000/seq.npy b_seq (1000000, 2, 2, )
    AB
    [sseq_index_ab::desc u.size 148849 opt BRIEF mode 6sseq_index_ab_chi2::desc sum  1825.2743 ndf 1805.0000 sum/ndf     1.0112 sseq_index_ab_chi2_ABSUM_MIN:40.0000
        TO AB                                                                                            :  126549 127024 :     0.8898 : Y :       2      3 :   
        TO BT BT BT BT BT BT SD                                                                          :   70552  70352 :     0.2839 : Y :      18      2 :   
        TO BT BT BT BT BT BT SA                                                                          :   57264  57599 :     0.9770 : Y :       5     29 :   
        TO SC AB                                                                                         :   51434  51389 :     0.0197 : Y :       4     13 :   
        TO SC BT BT BT BT BT BT SD                                                                       :   35993  36178 :     0.4742 : Y :      58     31 :   
        TO SC BT BT BT BT BT BT SA                                                                       :   29779  30082 :     1.5337 : Y :     124    135 :   
        TO SC SC AB                                                                                      :   19993  19624 :     3.4369 : Y :     137     20 :   
        TO RE AB                                                                                         :   18319  18271 :     0.0630 : Y :       9     56 :   
        TO SC SC BT BT BT BT BT BT SD                                                                    :   15499  15405 :     0.2859 : Y :      19     10 :   
        TO BT BT SA                                                                                      :   14137  14176 :     0.0537 : Y :     205     14 :   
        TO SC SC BT BT BT BT BT BT SA                                                                    :   12842  12942 :     0.3878 : Y :      24     17 :   
        TO BT BT AB                                                                                      :   10587  10493 :     0.4192 : Y :      72    233 :   
        TO BT AB                                                                                         :    9534   9349 :     1.8125 : Y :      36    242 :   
        TO SC SC SC AB                                                                                   :    7544   7482 :     0.2558 : Y :      90    112 :   
        TO RE BT BT BT BT BT BT SD                                                                       :    7439   7395 :     0.1305 : Y :     197    114 :   
        TO RE BT BT BT BT BT BT SA                                                                       :    7154   7023 :     1.2105 : Y :      48    245 :   
        TO SC RE AB                                                                                      :    7137   7001 :     1.3082 : Y :     110    102 :   
        TO BT BT BT BT BT BT BT SR SA                                                                    :    6375   6380 :     0.0020 : Y :      16     32 :   
        TO SC SC SC BT BT BT BT BT BT SD                                                                 :    6179   6243 :     0.3297 : Y :     145     12 :   
        TO BT BT BT BT SA                                                                                :    6231   6243 :     0.0115 : Y :      55      7 :   
        TO SC BT BT AB                                                                                   :    6177   6210 :     0.0879 : Y :     153     59 :   
        TO BT BT BT BT SD                                                                                :    6147   5976 :     2.4120 : Y :      13      6 :   
        TO SC BT AB                                                                                      :    5774   5775 :     0.0001 : Y :       8     37 :   
        TO BT BT BT SA                                                                                   :    5666   5406 :     6.1055 : Y :      71     79 :   
        TO RE RE AB                                                                                      :    5539   5455 :     0.6418 : Y :     267     84 :   
        TO BT BT DR BT SA                                                                                :    5456   5428 :     0.0720 : Y :     600    378 :   
        TO SC SC SC BT BT BT BT BT BT SA                                                                 :    5102   5097 :     0.0025 : Y :      23    259 :   
        TO BT BT BT BT BT BT BT SA                                                                       :    5063   4965 :     0.9577 : Y :     176    168 :   
        TO SC BT BT SA                                                                                   :    3819   3949 :     2.1756 : Y :     120    440 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                               :    3805   3937 :     2.2506 : Y :     362    300 :   
        TO BT BT BT BT AB                                                                                :    3834   3697 :     2.4922 : Y :     225   1555 :   
        TO RE SC AB                                                                                      :    3660   3548 :     1.7403 : Y :      54    239 :   
        TO SC RE BT BT BT BT BT BT SD                                                                    :    3205   3087 :     2.2130 : Y :     292     11 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                               :    3122   3194 :     0.8208 : Y :      22    199 :   
        TO SC BT BT BT BT BT BT BT SR SA                                                                 :    3155   3181 :     0.1067 : Y :     243     41 :   
        TO SC BT BT BT BT SA                                                                             :    3091   3124 :     0.1752 : Y :     536    187 :   
        TO BT BT BT BT BT BT BT SR SR SA                                                                 :    3049   3106 :     0.5279 : Y :     286    304 :   
        TO BT BT BT BT BT BT BT BT SD                                                                    :    3089   3089 :     0.0000 : Y :     181    849 :   
        TO SC BT BT BT SA                                                                                :    3044   3015 :     0.1388 : Y :     121    179 :   
        TO SC RE BT BT BT BT BT BT SA                                                                    :    2889   3003 :     2.2057 : Y :     151    598 :   
        TO SC BT BT BT BT BT BT BT SA                                                                    :    2784   2872 :     1.3692 : Y :      76     97 :   
        TO SC SC BT BT AB                                                                                :    2789   2854 :     0.7487 : Y :     636    278 :   
        TO SC BT BT BT BT SD                                                                             :    2843   2756 :     1.3518 : Y :     224    444 :   
        TO RE SC BT BT BT BT BT BT SD                                                                    :    2838   2775 :     0.7071 : Y :     282    381 :   
        TO SC SC BT AB                                                                                   :    2792   2662 :     3.0986 : Y :     987     58 :   
        TO SC SC SC SC AB                                                                                :    2745   2727 :     0.0592 : Y :     142   1312 :   
        TO BT BT BT BT BT BT BT BT SA                                                                    :    2683   2673 :     0.0187 : Y :     621     74 :   
        TO SC SC RE AB                                                                                   :    2626   2683 :     0.6120 : Y :     445    421 :   
        TO SC BT BT BT BT BT BT BT BT SD                                                                 :    2658   2610 :     0.4374 : Y :     102    533 :   
        TO RE SC BT BT BT BT BT BT SA                                                                    :    2632   2619 :     0.0322 : Y :     781    125 :   
        TO SC SC SC SC BT BT BT BT BT BT SD                                                              :    2345   2353 :     0.0136 : Y :      59    301 :   
        TO RE RE BT BT BT BT BT BT SD                                                                    :    2240   2166 :     1.2429 : Y :     655   1477 :   
        TO SC BT BT BT BT BT BT BT BT SA                                                                 :    2217   2200 :     0.0654 : Y :     534    130 :   
        TO BT BT BT BT BT BT BT SR SR SR SA                                                              :    2208   2191 :     0.0657 : Y :     528   1142 :   
        TO RE RE BT BT BT BT BT BT SA                                                                    :    2190   2130 :     0.8333 : Y :    1501     73 :   
        TO SC RE RE AB                                                                                   :    2118   2085 :     0.2591 : Y :    1340    493 :   
        TO SC SC SC SC BT BT BT BT BT BT SA                                                              :    1949   1942 :     0.0126 : Y :     525    254 :   
        TO SC BT BT BT BT AB                                                                             :    1842   1897 :     0.8090 : Y :    1667    267 :   
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                            :    1804   1804 :     0.0000 : Y :     448    413 :   
        TO RE RE RE AB                                                                                   :    1610   1651 :     0.5155 : Y :    1230    415 :   
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 148849 opt AZERO mode 1
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                            :      -1     14 :     0.0000 : N :      -1 107090 : AZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR BT SD                                         :      -1     13 :     0.0000 : N :      -1  42426 : AZERO C2EXC  
        TO BT BT SR BT BT AB                                                                             :      -1     12 :     0.0000 : N :      -1  15449 : AZERO C2EXC  
        TO BT BT DR BT BT BT SR BT SD                                                                    :      -1     11 :     0.0000 : N :      -1 167121 : AZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 148849 opt BZERO mode 2
        TO BT BT DR BT BT BT SD                                                                          :      26     -1 :     0.0000 : N :    1930     -1 : BZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BT SD                                            :      21     -1 :     0.0000 : N :   10972     -1 : BZERO C2EXC  
        TO BT BT BT BT BT DR BT SA                                                                       :      19     -1 :     0.0000 : N :   48549     -1 : BZERO C2EXC  
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                               :      15     -1 :     0.0000 : N :   12882     -1 : BZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BT SD                                         :      13     -1 :     0.0000 : N :   54864     -1 : BZERO C2EXC  
        TO SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT SR SD                                            :      13     -1 :     0.0000 : N :    5370     -1 : BZERO C2EXC  
    ]sseq_index_ab::desc

    AB
    [sseq_index_ab::desc u.size 148849 opt DEVIANT mode 5
    :r:`TO BT BT BT AB                                                                                   :     536    648 :    10.5946 : Y :     417     42 : DEVIANT  `
    :r:`TO BT BT BT BT BR BR BR DR AB                                                                    :       5     36 :    23.4390 : Y :  185265  21886 : DEVIANT  `
    ]sseq_index_ab::desc

    f

    CMDLINE:/data/blyth/junotop/opticks/sysrap/tests/sseq_index_test.py
    f.base:/data/blyth/opticks/sseq_index_test

      : f.sseq_index_ab_chi2                               :                 (4,) : 0:00:00.951428 

     min_stamp : 2024-11-14 18:22:17.397731 
     max_stamp : 2024-11-14 18:22:17.397731 
     dif_stamp : 0:00:00 
     age_stamp : 0:00:00.951428 
    [1825.274 1805.      40.       0.   ]
    c2sum/c2n:c2per(C2CUT)  1825.27/1805:1.011 (40) pv[1.000,> 0.05 : null-hyp ] 
    c2sum :  1825.2743 c2n :  1805.0000 c2per:     1.0112  C2CUT:   40 
    P[blyth@localhost ALL0]$ 





With hierarchy fix (but before the adhoc transform flip) : deviation bigger and to the other side (Opticks less) due to unexpected "TO BT SA" being higher
--------------------------------------------------------------------------------------------------------------------------------------------------------------

* A:Opticks reduced a lot in "TO BT BT SA" 
* B:Geant4 almost unchanged 
* "TO BT BT SA" deviation is bigger
* now a much bigger deviation shows up "TO BT SA"

Subsequent simtrace reveals this check was with Fastener injection into AdditionAcrylic with inverted 
radial shift wrong : so the Fasteners were poking into the Acrylic. 

::

    a_path $AFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/A000/seq.npy a_seq (1000000, 2, 2, )
    b_path $BFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/B000/seq.npy b_seq (1000000, 2, 2, )
    AB
    [sseq_index_ab::desc u.size 152262 opt BRIEF mode 6sseq_index_ab_chi2::desc sum  9998.5956 ndf 1825.0000 sum/ndf     5.4787 sseq_index_ab_chi2_ABSUM_MIN:40.0000
        TO AB                                                                                            :  126549 126567 :     0.0013 : Y :       2      3 :   
        TO BT BT BT BT BT BT SD                                                                          :   70475  70748 :     0.5277 : Y :      18      1 :   
        TO BT BT BT BT BT BT SA                                                                          :   57091  56883 :     0.3796 : Y :       5      8 :   
        TO SC AB                                                                                         :   51434  51320 :     0.1265 : Y :       4     12 :   
        TO SC BT BT BT BT BT BT SD                                                                       :   35876  35757 :     0.1977 : Y :      58     11 :   
        TO SC BT BT BT BT BT BT SA                                                                       :   29661  29875 :     0.7692 : Y :     124     22 :   
        TO SC SC AB                                                                                      :   19993  20115 :     0.3711 : Y :     137     57 :   

    :r:`TO BT BT SA                                                                                      :   15997  18574 :   192.0954 : Y :     205    118 : DEVIANT  `
        ##  A-B = -2577   

        TO RE AB                                                                                         :   18319  18519 :     1.0858 : Y :       9     41 :   
        TO SC SC BT BT BT BT BT BT SD                                                                    :   15451  15590 :     0.6224 : Y :      19     75 :   
        TO SC SC BT BT BT BT BT BT SA                                                                    :   12785  12972 :     1.3577 : Y :      24     35 :   
        TO BT BT AB                                                                                      :   10955  11153 :     1.7733 : Y :      72     31 :   
        TO BT AB                                                                                         :    9270   9271 :     0.0001 : Y :      36     26 :   
        TO SC SC SC AB                                                                                   :    7544   7472 :     0.3452 : Y :      90    162 :   
        TO BT BT BT BT BT BT BT SA                                                                       :    7435   7497 :     0.2574 : Y :     176     24 :   
        TO RE BT BT BT BT BT BT SD                                                                       :    7417   7491 :     0.3673 : Y :     197     34 :   
        TO SC RE AB                                                                                      :    7137   7135 :     0.0003 : Y :     110     17 :   
        TO RE BT BT BT BT BT BT SA                                                                       :    7124   7104 :     0.0281 : Y :      48     79 :   
        TO SC BT BT AB                                                                                   :    6374   6401 :     0.0571 : Y :     153     59 :   
        TO BT BT BT BT BT BT BT SR SA                                                                    :    6375   6323 :     0.2129 : Y :      16     56 :   
        TO BT BT BT BT SD                                                                                :    6147   6135 :     0.0117 : Y :      13    285 :   
        TO SC SC SC BT BT BT BT BT BT SD                                                                 :    6146   6134 :     0.0117 : Y :     145     64 :   

    :r:`TO SC BT BT SA                                                                                   :    4979   6119 :   117.1022 : Y :     120      5 : DEVIANT  `

        TO SC BT AB                                                                                      :    5600   5933 :     9.6149 : Y :       8    147 :   
        TO BT BT DR BT SA                                                                                :    5447   5546 :     0.8916 : Y :     600    161 :   
        TO RE RE AB                                                                                      :    5539   5323 :     4.2953 : Y :     267    119 :   
        TO BT BT BT SA                                                                                   :    5298   5316 :     0.0305 : Y :     745    192 :   
        TO SC SC SC BT BT BT BT BT BT SA                                                                 :    5084   4999 :     0.7166 : Y :      23     15 :   
        TO SC BT BT BT BT BT BT BT SA                                                                    :    4416   4476 :     0.4049 : Y :      20    421 :   

    :r:`TO BT SA                                                                                         :    3828    168 :  3352.2523 : Y :      71   2700 : DEVIANT  `

        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                               :    3805   3760 :     0.2677 : Y :     362    107 :   
        TO RE SC AB                                                                                      :    3660   3588 :     0.7152 : Y :      54     55 :   
        TO SC BT BT BT BT BT BT BT SR SA                                                                 :    3153   3291 :     2.9553 : Y :     243    639 :   
        TO SC RE BT BT BT BT BT BT SD                                                                    :    3190   3123 :     0.7111 : Y :     292    365 :   
        TO BT BT BT BT BT BT BT SD                                                                       :    3129   3145 :     0.0408 : Y :     181     74 :   
        TO BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                               :    3138   3141 :     0.0014 : Y :      22    712 :   
        TO BT BT BT BT BT BT BT SR SR SA                                                                 :    3049   2954 :     1.5034 : Y :     286    444 :   




Look into "TO BT SA" with LSExpDetectorConstruction__setupCD_Sticks_Fastener_CONFIG=1 
------------------------------------------------------------------------------------------

:: 

    HSEL="TO BT SA" PICK=AB ~/o/G4CXTest_GEOM.sh ana 

    ra.shape (3828, 32, 4, 4) 
    rb.shape (168, 32, 4, 4) 

    u_lbnd_ra[ 0] 108   n_lbnd_ra[ 0]    3822   cf.sim.bndnamedict.get(108) : Acrylic/Implicit_RINDEX_NoRINDEX_lAddition_phys_lFasteners_phys//Steel 
    u_lbnd_ra[ 1] 125   n_lbnd_ra[ 1]       6   cf.sim.bndnamedict.get(125) : Vacuum/NNVTMCPPMT_PMT_20inch_mcp_plate_opsurface//Steel 


Almost all the "TO BT SA" deviant are onto the Acrylic/Implicit_RINDEX_NoRINDEX_lAddition_phys_lFasteners_phys//Steel

* NB this did not exit the Acrylic : so indicates the AdditionAcrylic is coincident? with the Acrylic sphere 

  * this is not a long RT its tracing from within the Acrylic sphere and not hitting the otherside



laptop pyvista plotting
-------------------------

3D plotting that history : clumps onto sticks vaguely apparent::


   ~/o/G4CXTest_GEOM.sh gevt

   PICK=AB HSEL="TO BT BT SA" SEL=0 ~/o/G4CXTest_GEOM.sh dna  


After heirarchy fix the deviant "TO BT SA" is obviously from the "IonRing" of fastener::

   PICK=AB HSEL="TO BT SA" SEL=0 ~/o/G4CXTest_GEOM.sh dna  


Review AdditionAcrylicConstruction::m_simplify_csg  --additionacrylic-simplify-csg
--------------------------------------------------------------------------------------

This is just not doing subtraction of cavities for the fastener


Is the cause of the "TO BT SA" the coincidence of AdditionAcrylic and the Acrylic sphere ?
--------------------------------------------------------------------------------------------

Add an rdelta to check this::

    export AdditionAcrylicConstruction__rdelta_mm=1


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



::

    LSExpDetectorConstruction::setupCD_Sticks_Fastener  addition_PosR 17824 fastener_PosR 17844 fastener_dR 20 addition_lv YES fastener_lv YES



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


With fixed heirarchy, dont get the unexpected boundary::

    ra.shape (15997, 32, 4, 4) 
    rb.shape (18574, 32, 4, 4) 
     u_lbnd_ra[ 0]  99   n_lbnd_ra[ 0]   14137   cf.sim.bndnamedict.get( 99) : Tyvek//CDInnerTyvekSurface/Water 
     u_lbnd_ra[ 1] 107   n_lbnd_ra[ 1]    1860   cf.sim.bndnamedict.get(107) : Water/Implicit_RINDEX_NoRINDEX_pInnerWater_lUpper_phys//Steel 




Expected the Steel to be within Acrylic not Water
---------------------------------------------------

Look into this over in ~/j/setupCD_Sticks_Fastener/Fastener_asis_sibling_soup.rst


