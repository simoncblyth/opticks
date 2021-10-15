CSGOptiXSimulate_render_problem_geometry
============================================


cx::

    ./b7.sh 

cxs0.sh::

    ...
    cxs=${CXS:-1}   # collect sets of config underneath CXS

    if [ "$cxs" == "1" ]; then
        moi=Hama
        cegs=16:0:9:1000:18700:0:0:100

    elif [ "$cxs" == "2" ]; then
        moi=uni_acrylic3
        cegs=16:0:9:1000
    fi 

    export MOI=${MOI:-$moi}
    export CEGS=${CEGS:-$cegs}
    export CXS=${CXS:-$cxs}

    CSGOptiXSimulate




MOI Hama::

    cx ; CXS=1 ./cxs0.sh 

    2021-10-15 00:22:26.808 INFO  [185512] [CSGOptiX::setTop@158] ] tspec i0
    2021-10-15 00:22:26.808 INFO  [185512] [main@84]  moi Hama midx 104 mord 0 iidx 0
    2021-10-15 00:22:26.809 INFO  [185512] [CSGTarget::getGlobalCenterExtent@130]  repeatIdx 3 primIdx 1 inst.size 5000 iidx 0 iidx_in_range 1 local_ce ( 0.000, 0.000,-132.450,332.450) 
    2021-10-15 00:22:26.809 INFO  [185512] [CSGTarget::getGlobalCenterExtent@151]  q (-1.000, 0.000,-0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000,-1.000, 0.000) (-492.566,-797.087,19365.000, 1.000)  ins_idx 38213 ias_idx 0
    2021-10-15 00:22:26.809 INFO  [185512] [CSGTarget::getGlobalCenterExtent@163]  gpr CSGPrim numNode/node/tran/plan   7 17384 8009    0 sbtOffset/meshIdx/repeatIdx/primIdx    1  104    3    1 mn (-756.566,-1061.087,19165.000)  mx (-228.566,-533.087,19829.900)  gce (-492.566,-797.087,19497.449,332.450) 
    2021-10-15 00:22:26.809 INFO  [185512] [main@86]  rc 0 MOI.ce (-492.566 -797.087 19497.4 332.45)
    2021-10-15 00:22:26.810 INFO  [185512] [main@109] override the MOI.ce with CEGS.ce (18700 0 0 100)
    2021-10-15 00:22:26.810 INFO  [185512] [main@112]  CEGS nx:ny:nz:photons_per_genstep 16:0:9:1000
     nx 16 ny 0 nz 9 gs 627
    2021-10-15 00:22:26.810 INFO  [185512] [CSGOptiX::setCE@222]  ce [ 18700 0 0 100] tmin_model 0.1 tmin 10
    2021-10-15 00:22:26.810 INFO  [185512] [Composition::setNear@2424]  intended 10 result 10
    2021-10-15 00:22:26.810 INFO  [185512] [QEvent::setGensteps@115]  num_gs 627


MOI uni_acrylic3 leads to very big extent 26234.6::

    cx ; CXS=2 ./cxs0.sh 
    ...

    2021-10-14 23:52:29.717 INFO  [138433] [SBT::createGeom@101] ]
    2021-10-14 23:52:29.717 INFO  [138433] [CSGOptiX::init@120] ]
    2021-10-14 23:52:29.717 INFO  [138433] [CSGOptiX::setTop@148] [ tspec i0
    2021-10-14 23:52:29.718 INFO  [138433] [SBT::getAS@523]  spec i0 c i idx 0
    2021-10-14 23:52:29.718 INFO  [138433] [CSGOptiX::setTop@158] ] tspec i0
    2021-10-14 23:52:29.718 INFO  [138433] [main@84]  moi uni_acrylic3 midx 96 mord 0 iidx 0
    2021-10-14 23:52:29.718 INFO  [138433] [CSGTarget::getGlobalCenterExtent@130]  
       repeatIdx 8 
       primIdx 0 
       inst.size 590 iidx 0 iidx_in_range 1 
       local_ce ( 0.000, 0.000,17750.000,17890.000) 

    2021-10-14 23:52:29.718 INFO  [138433] [CSGTarget::getGlobalCenterExtent@151]  q (-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)  ins_idx 47383 ias_idx 0
    2021-10-14 23:52:29.718 INFO  [138433] [CSGTarget::getGlobalCenterExtent@163]  gpr CSGPrim numNode/node/tran/plan  31 17496 8062    0 sbtOffset/meshIdx/repeatIdx/primIdx    0   96    8    0 
      mn (-25863.145,-26229.033,-19473.879)  
      mx (25871.172,26240.086,19613.213)  
     gce ( 4.014, 5.526,69.667,26234.559) 

    2021-10-14 23:52:29.718 INFO  [138433] [main@86]  rc 0 MOI.ce (4.01367 5.52637 69.667 26234.6)
    2021-10-14 23:52:29.719 INFO  [138433] [main@112]  CEGS nx:ny:nz:photons_per_genstep 16:0:9:1000
     nx 16 ny 0 nz 9 gs 627
    2021-10-14 23:52:29.719 INFO  [138433] [CSGOptiX::setCE@222]  ce [ 4.01367 5.52637 69.667 26234.6] tmin_model 0.1 tmin 2623.46
    2021-10-14 23:52:29.719 INFO  [138433] [Composition::setNear@2424]  intended 2623.46 result 2623.46



CSGPrimTest::

     primIdx       3096 pr_primIdx          1 gasIdx          3 meshIdx        104 meshName                               HamamatsuR12860sMask ce (      0.00,      0.00,   -132.45,    332.45)
     primIdx       3097 pr_primIdx          2 gasIdx          3 meshIdx        108 meshName           HamamatsuR12860_PMT_20inch_pmt_solid_1_9 ce (      0.00,      0.00,   -130.00,    320.00)
     primIdx       3098 pr_primIdx          3 gasIdx          3 meshIdx        107 meshName          HamamatsuR12860_PMT_20inch_body_solid_1_9 ce (      0.00,      0.00,   -130.00,    320.00)
     primIdx       3099 pr_primIdx          4 gasIdx          3 meshIdx        105 meshName          HamamatsuR12860_PMT_20inch_inner1_solid_I ce (      0.00,      0.00,     92.50,    249.00)
     primIdx       3100 pr_primIdx          5 gasIdx          3 meshIdx        106 meshName        HamamatsuR12860_PMT_20inch_inner2_solid_1_9 ce (      0.00,      0.00,   -172.50,    272.50)
     primIdx       3101 pr_primIdx          0 gasIdx          4 meshIdx        121 meshName                          mask_PMT_20inch_vetosMask ce (      0.00,      0.00,     84.55,    264.00)
     primIdx       3102 pr_primIdx          1 gasIdx          4 meshIdx        125 meshName                      PMT_20inch_veto_pmt_solid_1_2 ce (      0.00,      0.00,     87.00,    254.00)
     primIdx       3103 pr_primIdx          2 gasIdx          4 meshIdx        122 meshName                       PMT_20inch_veto_inner1_solid ce (      0.00,      0.00,     42.00,    254.00)
     primIdx       3104 pr_primIdx          3 gasIdx          4 meshIdx        123 meshName                       PMT_20inch_veto_inner2_solid ce (      0.00,      0.00,     42.00,    254.00)
     primIdx       3105 pr_primIdx          0 gasIdx          5 meshIdx         93 meshName                                     sStrutBallhead ce (      0.00,      0.00,      0.00,     50.00)
     primIdx       3106 pr_primIdx          0 gasIdx          6 meshIdx         94 meshName                                               uni1 ce (      0.00,      0.00,     -7.50,    206.20)
     primIdx       3107 pr_primIdx          0 gasIdx          7 meshIdx         95 meshName                                         base_steel ce (      0.00,      0.00,    -50.50,    195.00)

     primIdx       3108 pr_primIdx          0 gasIdx          8 meshIdx         96 meshName                                       uni_acrylic3 ce (      0.00,      0.00,  17750.00,  17890.00)

     primIdx       3109 pr_primIdx          0 gasIdx          9 meshIdx          7 meshName                                             sPanel ce (      0.00,      0.00,      0.00,   3430.60)
     primIdx       3110 pr_primIdx          1 gasIdx          9 meshIdx          6 meshName                                         sPanelTape ce (      0.00,      0.00,      0.00,   3430.00)
     primIdx       3111 pr_primIdx          2 gasIdx          9 meshIdx          5 meshName                                               sBar ce (      0.00,   -831.60,      0.00,   3430.00)



Looks like the subtracted acrylic sphere feeds into making ginormous bbox.  How to fix that ?


CSG_GGeo_Convert::convertSolid::

    210     
    211     AABB bb = {} ;
    212     
    213     // over the "layers/volumes" of the solid (eg multiple vols of PMT) 
    214     for(unsigned primIdx=0 ; primIdx < numPrim ; primIdx++) 
    215     {   
    216         unsigned globalPrimIdx = so->primOffset + primIdx ;
    217         unsigned globalPrimIdx_0 = foundry->getNumPrim() ; 
    218         assert( globalPrimIdx == globalPrimIdx_0 ); 
    219         
    220         CSGPrim* prim = convertPrim(comp, primIdx);
    221         bb.include_aabb( prim->AABB() );
    222 
    223         unsigned sbtIdx = prim->sbtIndexOffset() ;
    224         //assert( sbtIdx == globalPrimIdx  );  
    225         assert( sbtIdx == primIdx  );
    226 
    227         prim->setRepeatIdx(repeatIdx);
    228         prim->setPrimIdx(primIdx);
    229         //LOG(info) << prim->desc() ;
    230     }   
    231     so->center_extent = bb.center_extent() ;



31 node complete binary tree::


     .                                                                      1(0)
                                                                             in
                                           10(1)                                                                       11(2)
                                            in                                                                           in
                            100(3)                  101(4)                                            110(5)                           111(6)
                               in                       in                                               in                               un
                  1000(7)        1001(8)      1010(9)          1011(10)                    1100(11)           1101(12)        1110(13)          1111(14)
                    in            in            in                in                          cy                un                cy                  cy
           10000     10001  10010  10011   10100  10101   10110     10111               11000  11001      11010   11011     11100    11101      11110      11111
     (-1)   (15)     (16)   (17)   (18)    (19)   (20)    (21)      (22)                (23)   (24)       (25)    (26)       (27)     (28)        (29)       (30)  
             sp       cy     cy     cy      cy     cy      cy        cy                  ze     ze          co      cy         ze       ze          ze         ze


Remember to +1 when associating nodes with the above complete binary tree::

    2021-10-14 20:40:13.063 ERROR [3061771] [CSG_GGeo_Convert::convertAllSolid@112] proceeding with convert for repeatIdx 8
    2021-10-14 20:40:13.063 INFO  [3061771] [*CSG_GGeo_Convert::convertSolid@210]  repeatIdx 8 nmm 10 numPrim(GParts.getNumPrim) 1 rlabel r8 num_inst 590 dump_ridx 8 dump 1
    CSG_GGeo_Convert::convertPrim primIdx 0 numPrim 1 numParts 31 meshIdx 96 last_ridx 8 dump 1
    CSGNode     0 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     1 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     2 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     3 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     4 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     5 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     6 un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     7 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     8 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode     9 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode    10 in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode    11 cy aabb:   102.0  -130.0  -140.0   130.0  -102.0   -35.0  trIdx: 8063
    CSGNode    12 un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx: 0
    CSGNode    13 cy aabb:  -208.0  -208.0   -35.2   208.0   208.0    -4.8  trIdx: 8064
    CSGNode    14 cy aabb:  -120.0  -120.0   -35.4   120.0   120.0    -4.6  trIdx: 8065
    CSGNode    15 sp aabb: -17820.0 -17820.0     0.0 17820.0 17820.0 35640.0  trIdx: 8066
    CSGNode    16 cy aabb:   150.0   -14.0  -140.0   178.0    14.0   -35.0  trIdx: 8067
    CSGNode    17 cy aabb:   102.0   102.0  -140.0   130.0   130.0   -35.0  trIdx: 8068
    CSGNode    18 cy aabb:   -14.0   150.0  -140.0    14.0   178.0   -35.0  trIdx: 8069
    CSGNode    19 cy aabb:  -130.0   102.0  -140.0  -102.0   130.0   -35.0  trIdx: 8070
    CSGNode    20 cy aabb:  -178.0   -14.0  -140.0  -150.0    14.0   -35.0  trIdx: 8071
    CSGNode    21 cy aabb:  -130.0  -130.0  -140.0  -102.0  -102.0   -35.0  trIdx: 8072
    CSGNode    22 cy aabb:   -14.0  -178.0  -140.0    14.0  -150.0   -35.0  trIdx: 8073
    CSGNode    23 ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx: 0
    CSGNode    24 ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx: 0
    CSGNode    25 co aabb:  -450.0  -450.0  -140.0   450.0   450.0     1.0  trIdx: 8074
    CSGNode    26 cy aabb:  -450.0  -450.0     0.0   450.0   450.0     5.7  trIdx: 8075
    CSGNode    27 ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx: 0
    CSGNode    28 ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx: 0
    CSGNode    29 ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx: 0
    CSGNode    30 ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx: 0
    dump.Prim.AABB  -17820.00  -17820.00    -140.00   17820.00   17820.00   35640.00 
    2021-10-14 20:40:13.064 INFO  [3061771] [CSG_GGeo_Convert::addInstances@164]  reapeatIdx 8 iid 590,1,4
    2021-10-14 20:40:13.064 INFO  [3061771] [*CSG_GGeo_Convert::convertSolid@247]  solid.bb  [ (-17820.000,-17820.000,-140.000)  : (17820.000,17820.000,35640.000)  | (35640.000,35640.000,35780.000)  ] 
    2021-10-14 20:40:13.064 INFO  [3061771] [*CSG_GGeo_Convert::convertSolid@248]  solid.desc CSGSolid               r8 primNum/Offset     1 3108 ce ( 0.000, 0.000,17750.000,17890.000) 
    2021-10-14 20:40:13.064 ERROR [3061771] [CSG_GGeo_Convert::convertAllSolid@112] proceeding with convert for repeatIdx 9



Now try excluding bbox from complemented leaf nodes with only intersect in ancestry::


    2021-10-15 12:40:25.893 ERROR [3318882] [CSG_GGeo_Convert::convertAllSolid@112] proceeding with convert for repeatIdx 8
    2021-10-15 12:40:25.893 INFO  [3318882] [*CSG_GGeo_Convert::convertSolid@210]  repeatIdx 8 nmm 10 numPrim(GParts.getNumPrim) 1 rlabel r8 num_inst 590 dump_ridx 8 dump 1
    CSG_GGeo_Convert::convertPrim primIdx 0 numPrim 1 numParts 31 meshIdx 96 last_ridx 8 dump 1
      0 CSGNode     0  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 0 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
      1 CSGNode     1  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      2 CSGNode     2  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      3 CSGNode     3  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      4 CSGNode     4  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      5 CSGNode     5  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      6 CSGNode     6  un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      7 CSGNode     7  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      8 CSGNode     8  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      9 CSGNode     9  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     10 CSGNode    10  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     11 CSGNode    11 !cy aabb:   102.0  -130.0  -140.0   130.0  -102.0   -35.0  trIdx:  8063 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     12 CSGNode    12  un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     13 CSGNode    13 !cy aabb:  -208.0  -208.0   -35.2   208.0   208.0    -4.8  trIdx:  8064 atm 6 IsOnlyIntersectionMask 0 is_complemented_leaf 1 bbskip 0
     14 CSGNode    14  cy aabb:  -120.0  -120.0   -35.4   120.0   120.0    -4.6  trIdx:  8065 atm 6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     15 CSGNode    15 !sp aabb: -17820.0 -17820.0     0.0 17820.0 17820.0 35640.0  trIdx:  8066 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     16 CSGNode    16 !cy aabb:   150.0   -14.0  -140.0   178.0    14.0   -35.0  trIdx:  8067 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     17 CSGNode    17 !cy aabb:   102.0   102.0  -140.0   130.0   130.0   -35.0  trIdx:  8068 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     18 CSGNode    18 !cy aabb:   -14.0   150.0  -140.0    14.0   178.0   -35.0  trIdx:  8069 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     19 CSGNode    19 !cy aabb:  -130.0   102.0  -140.0  -102.0   130.0   -35.0  trIdx:  8070 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     20 CSGNode    20 !cy aabb:  -178.0   -14.0  -140.0  -150.0    14.0   -35.0  trIdx:  8071 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     21 CSGNode    21 !cy aabb:  -130.0  -130.0  -140.0  -102.0  -102.0   -35.0  trIdx:  8072 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     22 CSGNode    22 !cy aabb:   -14.0  -178.0  -140.0    14.0  -150.0   -35.0  trIdx:  8073 atm 4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     23 CSGNode    23  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm 4100 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     24 CSGNode    24  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm 4100 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     25 CSGNode    25  co aabb:  -450.0  -450.0  -140.0   450.0   450.0     1.0  trIdx:  8074 atm 6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     26 CSGNode    26  cy aabb:  -450.0  -450.0     0.0   450.0   450.0     5.7  trIdx:  8075 atm 6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     27 CSGNode    27  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm 4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     28 CSGNode    28  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm 4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     29 CSGNode    29  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm 4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     30 CSGNode    30  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm 4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
    dump.Prim.AABB    -450.00    -450.00    -140.00     450.00     450.00     **100.00**   
    2021-10-15 12:40:25.894 INFO  [3318882] [CSG_GGeo_Convert::addInstances@164]  reapeatIdx 8 iid 590,1,4


Page 40 of below presentation has a 2d figure illustrating cross section of geometry.

* http://localhost/env/presentation/juno_opticks_20210712.html
* https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210712.html

Notice that bbmax.z of 100.0 is wrong, seems the placeholder bbox from ze are not being excluded. 


After excluding those placeholders, get the expected bbox::

    2021-10-15 12:54:30.676 INFO  [3333474] [*CSG_GGeo_Convert::convertSolid@210]  repeatIdx 8 nmm 10 numPrim(GParts.getNumPrim) 1 rlabel r8 num_inst 590 dump_ridx 8 dump 1
    CSG_GGeo_Convert::convertPrim primIdx 0 numPrim 1 numParts 31 meshIdx 96 last_ridx 8 dump 1
      0 CSGNode     0  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     0 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
      1 CSGNode     1  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      2 CSGNode     2  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      3 CSGNode     3  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      4 CSGNode     4  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      5 CSGNode     5  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      6 CSGNode     6  un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      7 CSGNode     7  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      8 CSGNode     8  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
      9 CSGNode     9  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     10 CSGNode    10  in aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     11 CSGNode    11 !cy aabb:   102.0  -130.0  -140.0   130.0  -102.0   -35.0  trIdx:  8063 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     12 CSGNode    12  un aabb:    -0.0    -0.0    -0.0     0.0     0.0     0.0  trIdx:     0 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 0 bbskip 0
     13 CSGNode    13 !cy aabb:  -208.0  -208.0   -35.2   208.0   208.0    -4.8  trIdx:  8064 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 1 bbskip 0
     14 CSGNode    14  cy aabb:  -120.0  -120.0   -35.4   120.0   120.0    -4.6  trIdx:  8065 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     15 CSGNode    15 !sp aabb: -17820.0 -17820.0     0.0 17820.0 17820.0 35640.0  trIdx:  8066 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     16 CSGNode    16 !cy aabb:   150.0   -14.0  -140.0   178.0    14.0   -35.0  trIdx:  8067 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     17 CSGNode    17 !cy aabb:   102.0   102.0  -140.0   130.0   130.0   -35.0  trIdx:  8068 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     18 CSGNode    18 !cy aabb:   -14.0   150.0  -140.0    14.0   178.0   -35.0  trIdx:  8069 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     19 CSGNode    19 !cy aabb:  -130.0   102.0  -140.0  -102.0   130.0   -35.0  trIdx:  8070 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     20 CSGNode    20 !cy aabb:  -178.0   -14.0  -140.0  -150.0    14.0   -35.0  trIdx:  8071 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     21 CSGNode    21 !cy aabb:  -130.0  -130.0  -140.0  -102.0  -102.0   -35.0  trIdx:  8072 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     22 CSGNode    22 !cy aabb:   -14.0  -178.0  -140.0    14.0  -150.0   -35.0  trIdx:  8073 atm     4 IsOnlyIntersectionMask 1 is_complemented_leaf 1 bbskip 1
     23 CSGNode    23  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm  4100 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 1
     24 CSGNode    24  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm  4100 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 1
     25 CSGNode    25  co aabb:  -450.0  -450.0  -140.0   450.0   450.0     1.0  trIdx:  8074 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     26 CSGNode    26  cy aabb:  -450.0  -450.0     0.0   450.0   450.0     5.7  trIdx:  8075 atm     6 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 0
     27 CSGNode    27  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm  4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 1
     28 CSGNode    28  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm  4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 1
     29 CSGNode    29  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm  4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 1
     30 CSGNode    30  ze aabb:  -100.0  -100.0  -100.0   100.0   100.0   100.0  trIdx:     0 atm  4102 IsOnlyIntersectionMask 0 is_complemented_leaf 0 bbskip 1
    dump.Prim.AABB    -450.00    -450.00    -140.00     450.00     450.00       5.70 
    2021-10-15 12:54:30.677 INFO  [3333474] [CSG_GGeo_Convert::addInstances@164]  reapeatIdx 8 iid 590,1,4
    2021-10-15 12:54:30.677 INFO  [3333474] [*CSG_GGeo_Convert::convertSolid@247]  solid.bb  [ (-450.000,-450.000,-140.000)  : (450.000,450.000, 5.700)  | (900.000,900.000,145.700)  ] 




