j1707 GTreeCheck missed lSteel from instancing
================================================


Hmm, vertex_min dont make much sense 

* it dont matter if its low poly or not if you have a large number of em you want to instance it.
* recall GTreeCheck is for triangulated, NScene::find_repeat_candidates does similar thing for analytic.



Old GTreeCheck reporting
---------------------------

::

    op --j1707 -G --debugger


    2017-08-17 12:33:00.159 INFO  [183431] [*GScintillatorLib::createBuffer@102] GScintillatorLib::createBuffer  ni 1 nj 4096 nk 1
    2017-08-17 12:33:00.159 FATAL [183431] [*GScintillatorLib::constructInvertedReemissionCDF@170] GScintillatorLib::constructInvertedReemissionCDF  was expecting to trim 2 values  l_srrd 33 l_rrd 39
    2017-08-17 12:33:00.159 INFO  [183431] [GPropertyLib::close@384] GPropertyLib::close type GScintillatorLib buf 1,4096,1
    2017-08-17 12:33:10.310 INFO  [183431] [GTreeCheck::findRepeatCandidates@161] GTreeCheck::findRepeatCandidates nall 35 repeat_min 120 vertex_min 250 candidates marked with ** 

    2017-08-17 12:33:10.328 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   0 pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 nvert    802 n PMT_3inch_log0x1c9ef80
    2017-08-17 12:33:10.346 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   1 pdig 1f0e3dad99908345f7439f8ffabdffc4 ndig  36572 nprog      0 nvert     50 n PMT_3inch_cntr_log0x1c9f1f0
    2017-08-17 12:33:10.366 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   2 pdig 70efdf2ec9b086079795c442636b55fb ndig  36572 nprog      0 nvert    146 n PMT_3inch_inner2_log0x1c9f120
    2017-08-17 12:33:10.386 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   3 pdig c74d97b01eae257e44aa9d5bade97baf ndig  36572 nprog      0 nvert    122 n PMT_3inch_inner1_log0x1c9f050
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   4 pdig 873d395b9e0f186e0a9369ced7e84293 ndig  36572 nprog      2 nvert    486 n PMT_3inch_body_log0x1c9eef0

    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   5 pdig d3d9446802a44259755d38e6d163e820 ndig  17739 nprog      0 nvert    484 n lMask0x18170e0
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   6 pdig c20ad4d76fe97759aa27a0c99bff6710 ndig  17739 nprog      0 nvert    482 n PMT_20inch_inner2_log0x1863310
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i   7 pdig 6512bd43d9caa6e02c990b0a82652dca ndig  17739 nprog      0 nvert    194 n PMT_20inch_inner1_log0x1863280
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   8 pdig 511dd06bf687b1e989d4ac84e25bc0a3 ndig  17739 nprog      2 nvert   1254 n PMT_20inch_body_log0x1863160
    2017-08-17 12:33:10.406 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i   9 pdig fd99bf5972e7592724cbd49bfb448953 ndig  17739 nprog      3 nvert   1832 n PMT_20inch_log0x18631f0
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i  10 pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 nvert   2366 n lMaskVirtual0x1816910

    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192] ** i  11 pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 nvert    914 n lFasteners0x1506370
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i  12 pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 nvert     96 n lSteel0x14dde40
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i  13 pdig 1679091c5a880faf6fb5e6087eb1b2dc ndig      1 nprog      0 nvert    362 n lTarget0x14dd830
    2017-08-17 12:33:10.407 INFO  [183431] [GTreeCheck::findRepeatCandidates@192]    i  14 pdig 3316c24dd38f0f2ca5d7250814b99d1a ndig      1 nprog      1 nvert    724 n lAcrylic0x14dd290

    2017-08-17 12:33:10.935 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig 873d395b9e0f186e0a9369ced7e84293 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.935 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig d3d9446802a44259755d38e6d163e820 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig c20ad4d76fe97759aa27a0c99bff6710 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig 511dd06bf687b1e989d4ac84e25bc0a3 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::operator@220] GTreeCheck::operator()  pdig fd99bf5972e7592724cbd49bfb448953 disallowd as isContainedRepeat 
    2017-08-17 12:33:10.936 INFO  [183431] [GTreeCheck::dumpRepeatCandidates@255] GTreeCheck::dumpRepeatCandidates 

     pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 placements  36572 n PMT_3inch_log0x1c9ef80
     pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 placements  17739 n lMaskVirtual0x1816910
     pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 placements    480 n lFasteners0x1506370
    2017-08-17 12:33:11.212 INFO  [183431] [GTreeCheck::labelTree@327] GTreeCheck::labelTree count of non-zero setRepeatIndex 289774
    2017-08-17 12:33:23.880 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 0 numPlacements 1 numSolids 290276
    2017-08-17 12:33:24.124 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 1 numPlacements 36572 numSolids 5
    2017-08-17 12:33:24.340 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 2 numPlacements 17739 numSolids 6
    2017-08-17 12:33:24.549 INFO  [183431] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@467] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 3 numPlacements 480 numSolids 1
    2017-08-17 12:33:24.549 INFO  [183431] [TimesTable::dump@103] Timer::dump filter: NONE
              0.000      t_absolute        t_delta
              0.716           0.716          0.716 : deltacheck
             10.192          10.909         10.192 : traverse
              0.144          11.053          0.144 : labelTree
             13.337          24.390         13.337 : makeMergedMeshAndInstancedBuffers
    2017-08-17 12:33:24.549 INFO  [183431] [GColorizer::traverse@93] GColorizer::traverse START




Improved GTreeCheck reporting, reveals missing lSteel repeater from vertex_min
---------------------------------------------------------------------------------

::

    2017-08-17 13:54:32.195 INFO  [208370] [GTreeCheck::operator@272] GTreeCheck::operator()  pdig 873d395b9e0f186e0a9369ced7e84293 disallowd as isContainedRepeat 
    2017-08-17 13:54:32.195 INFO  [208370] [GTreeCheck::operator@272] GTreeCheck::operator()  pdig d3d9446802a44259755d38e6d163e820 disallowd as isContainedRepeat 
    2017-08-17 13:54:32.196 INFO  [208370] [GTreeCheck::operator@272] GTreeCheck::operator()  pdig c20ad4d76fe97759aa27a0c99bff6710 disallowd as isContainedRepeat 
    2017-08-17 13:54:32.196 INFO  [208370] [GTreeCheck::operator@272] GTreeCheck::operator()  pdig 511dd06bf687b1e989d4ac84e25bc0a3 disallowd as isContainedRepeat 
    2017-08-17 13:54:32.196 INFO  [208370] [GTreeCheck::operator@272] GTreeCheck::operator()  pdig fd99bf5972e7592724cbd49bfb448953 disallowd as isContainedRepeat 
    2017-08-17 13:54:32.197 INFO  [208370] [GTreeCheck::findRepeatCandidates@247] GTreeCheck::findRepeatCandidates nall 35 repeat_min 120 vertex_min 250 cands 35 reps 3

     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 

     **  ##  idx   0 pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 nvert    802 n PMT_3inch_log0x1c9ef80
             idx   1 pdig 1f0e3dad99908345f7439f8ffabdffc4 ndig  36572 nprog      0 nvert     50 n PMT_3inch_cntr_log0x1c9f1f0
             idx   2 pdig 70efdf2ec9b086079795c442636b55fb ndig  36572 nprog      0 nvert    146 n PMT_3inch_inner2_log0x1c9f120
             idx   3 pdig c74d97b01eae257e44aa9d5bade97baf ndig  36572 nprog      0 nvert    122 n PMT_3inch_inner1_log0x1c9f050
     **      idx   4 pdig 873d395b9e0f186e0a9369ced7e84293 ndig  36572 nprog      2 nvert    486 n PMT_3inch_body_log0x1c9eef0

     **      idx   5 pdig d3d9446802a44259755d38e6d163e820 ndig  17739 nprog      0 nvert    484 n lMask0x18170e0
     **      idx   6 pdig c20ad4d76fe97759aa27a0c99bff6710 ndig  17739 nprog      0 nvert    482 n PMT_20inch_inner2_log0x1863310
             idx   7 pdig 6512bd43d9caa6e02c990b0a82652dca ndig  17739 nprog      0 nvert    194 n PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig 511dd06bf687b1e989d4ac84e25bc0a3 ndig  17739 nprog      2 nvert   1254 n PMT_20inch_body_log0x1863160
     **      idx   9 pdig fd99bf5972e7592724cbd49bfb448953 ndig  17739 nprog      3 nvert   1832 n PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 nvert   2366 n lMaskVirtual0x1816910

     **  ##  idx  11 pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 nvert    914 n lFasteners0x1506370

             idx  12 pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 nvert     96 n lSteel0x14dde40

     //// lSteel missed instancing from vertex_min cut        

             idx  13 pdig 1679091c5a880faf6fb5e6087eb1b2dc ndig      1 nprog      0 nvert    362 n lTarget0x14dd830
             idx  14 pdig 3316c24dd38f0f2ca5d7250814b99d1a ndig      1 nprog      1 nvert    724 n lAcrylic0x14dd290
             idx  15 pdig b415485a6aca3391ff9bc0cedefe5a39 ndig      1 nprog 290264 nvert 71789910 n lInnerWater0x14dccf0
             idx  16 pdig 035ccbc0331ee49145bc84efdcd42e60 ndig      1 nprog 290265 nvert 71790272 n lReflectorInCD0x14dc750
             idx  17 pdig ce0c2492a79afef1ff6f1e48a6b4934c ndig      1 nprog 290275 nvert 71790738 n lWorld0x14d9c00
             idx  18 pdig b5f9e1ae53ecdd2dd902cfc7a5a6bf2c ndig      1 nprog 290266 nvert 71790322 n lOuterWaterPool0x14dbd60
             idx  19 pdig 6899ebf7e701f7a850951c59547eb53c ndig      1 nprog 290267 nvert 71790372 n lPoolLining0x14db8b0
             idx  20 pdig 68cfdc5d3977ec2d38c1e74b71ad3d2e ndig      1 nprog 290268 nvert 71790422 n lBtmRock0x14db220
             idx  21 pdig c81e728d9d4c2f636f067f89cc14862c ndig      1 nprog      0 nvert     96 n lUpperChimneyTyvek0x2547c80
             idx  22 pdig c4ca4238a0b923820dcc509a6f75849b ndig      1 nprog      0 nvert     96 n lUpperChimneySteel0x2547bb0
             idx  23 pdig cfcd208495d565ef66e7dff9f98764da ndig      1 nprog      0 nvert     50 n lUpperChimneyLS0x2547ae0
             idx  24 pdig 746ea22d5a1acd3c4f37fe7d648e9767 ndig      1 nprog      3 nvert    292 n lUpperChimney0x2547a50
             idx  25 pdig a80eb4230c6c09bc0536ad95784c6f78 ndig      1 nprog      4 nvert    300 n lExpHall0x14da8d0
             idx  26 pdig 6d794eba2efcefa04607dadd5443354a ndig      1 nprog      5 nvert    308 n lTopRock0x14da5a0
             idx  27 pdig 1e6ef35822ab41ec6c862b0ee8686afa ndig      1 nprog      5 nvert   1602 n lLowerChimney0x254aa20
             idx  28 pdig 3c59dc048e8850243be8079a5c74d079 ndig      1 nprog      0 nvert    430 n lLowerChimneyTyvek0x254ab60
             idx  29 pdig b6d767d2f8ed5d21a44b0e5886680cb9 ndig      1 nprog      0 nvert     96 n lLowerChimneyAcrylic0x254ac30
             idx  30 pdig 37693cfc748049e45d87b8c7d8b9aacd ndig      1 nprog      0 nvert    700 n lLowerChimneySteel0x254ad00
             idx  31 pdig a4410129b0d223b22cbfe1aa3bb5f469 ndig      1 nprog      1 nvert    274 n lLowerChimneyLS0x254ad90
             idx  32 pdig 1ff1de774005f8da13f42943881c655f ndig      1 nprog      0 nvert    168 n lLowerChimneyBlocker0x254ae60
             idx  33 pdig 7ff5c8a0d84d0f85355630af10cf9176 ndig      1 nprog      1 nvert   1204 n lSurftube0x254b8d0
             idx  34 pdig 02e74f10e0327ad868d138f2b4fdd6f0 ndig      1 nprog      0 nvert    602 n lvacSurftube0x254ba90
    2017-08-17 13:54:32.197 INFO  [208370] [GTreeCheck::dumpRepeatCandidates@307] GTreeCheck::dumpRepeatCandidates 
     pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 placements  36572 n PMT_3inch_log0x1c9ef80
     pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 placements  17739 n lMaskVirtual0x1816910
     pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 placements    480 n lFasteners0x1506370
    2017-08-17 13:54:32.465 INFO  [208370] [GTreeCheck::labelTree@379] GTreeCheck::labelTree count of non-zero setRepeatIndex 289774
    2017-08-17 13:54:45.311 INFO  [208370] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@519] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 0 numPlacements 1 numSolids 290276
    2017-08-17 13:54:45.557 INFO  [208370] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@519] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 1 numPlacements 36572 numSolids 5
    2017-08-17 13:54:45.769 INFO  [208370] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@519] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 2 numPlacements 17739 numSolids 6
    2017-08-17 13:54:45.973 INFO  [208370] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@519] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 3 numPlacements 480 numSolids 1




Effectively remove vertex min cut for JUNO
----------------------------------------------

::

     409    if(resource->isJuno())
     410    {
     411        m_treecheck->setVertexMin(10);
     412        //m_treecheck->setVertexMin(250);
     413    }



::

    2017-08-17 14:05:30.125 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig 1f0e3dad99908345f7439f8ffabdffc4 disallowd as isContainedRepeat 
    2017-08-17 14:05:30.141 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig 70efdf2ec9b086079795c442636b55fb disallowd as isContainedRepeat 
    2017-08-17 14:05:30.159 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig c74d97b01eae257e44aa9d5bade97baf disallowd as isContainedRepeat 
    2017-08-17 14:05:30.176 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig 873d395b9e0f186e0a9369ced7e84293 disallowd as isContainedRepeat 
    2017-08-17 14:05:30.176 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig d3d9446802a44259755d38e6d163e820 disallowd as isContainedRepeat 
    2017-08-17 14:05:30.176 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig c20ad4d76fe97759aa27a0c99bff6710 disallowd as isContainedRepeat 
    2017-08-17 14:05:30.176 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig 6512bd43d9caa6e02c990b0a82652dca disallowd as isContainedRepeat 
    2017-08-17 14:05:30.176 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig 511dd06bf687b1e989d4ac84e25bc0a3 disallowd as isContainedRepeat 
    2017-08-17 14:05:30.177 INFO  [213429] [GTreeCheck::operator@270] GTreeCheck::operator()  pdig fd99bf5972e7592724cbd49bfb448953 disallowd as isContainedRepeat 
    2017-08-17 14:05:30.177 INFO  [213429] [GTreeCheck::findRepeatCandidates@245] GTreeCheck::findRepeatCandidates nall 35 repeat_min 120 vertex_min 10 cands 35 reps 4
     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 
     **  ##  idx   0 pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 nvert    802 n PMT_3inch_log0x1c9ef80
     **      idx   1 pdig 1f0e3dad99908345f7439f8ffabdffc4 ndig  36572 nprog      0 nvert     50 n PMT_3inch_cntr_log0x1c9f1f0
     **      idx   2 pdig 70efdf2ec9b086079795c442636b55fb ndig  36572 nprog      0 nvert    146 n PMT_3inch_inner2_log0x1c9f120
     **      idx   3 pdig c74d97b01eae257e44aa9d5bade97baf ndig  36572 nprog      0 nvert    122 n PMT_3inch_inner1_log0x1c9f050
     **      idx   4 pdig 873d395b9e0f186e0a9369ced7e84293 ndig  36572 nprog      2 nvert    486 n PMT_3inch_body_log0x1c9eef0

     **      idx   5 pdig d3d9446802a44259755d38e6d163e820 ndig  17739 nprog      0 nvert    484 n lMask0x18170e0
     **      idx   6 pdig c20ad4d76fe97759aa27a0c99bff6710 ndig  17739 nprog      0 nvert    482 n PMT_20inch_inner2_log0x1863310
     **      idx   7 pdig 6512bd43d9caa6e02c990b0a82652dca ndig  17739 nprog      0 nvert    194 n PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig 511dd06bf687b1e989d4ac84e25bc0a3 ndig  17739 nprog      2 nvert   1254 n PMT_20inch_body_log0x1863160
     **      idx   9 pdig fd99bf5972e7592724cbd49bfb448953 ndig  17739 nprog      3 nvert   1832 n PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 nvert   2366 n lMaskVirtual0x1816910

     **  ##  idx  11 pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 nvert    914 n lFasteners0x1506370

     **  ##  idx  12 pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 nvert     96 n lSteel0x14dde40

             idx  13 pdig 1679091c5a880faf6fb5e6087eb1b2dc ndig      1 nprog      0 nvert    362 n lTarget0x14dd830
             idx  14 pdig 3316c24dd38f0f2ca5d7250814b99d1a ndig      1 nprog      1 nvert    724 n lAcrylic0x14dd290
             idx  15 pdig b415485a6aca3391ff9bc0cedefe5a39 ndig      1 nprog 290264 nvert 71789910 n lInnerWater0x14dccf0
             idx  16 pdig 035ccbc0331ee49145bc84efdcd42e60 ndig      1 nprog 290265 nvert 71790272 n lReflectorInCD0x14dc750
             idx  17 pdig ce0c2492a79afef1ff6f1e48a6b4934c ndig      1 nprog 290275 nvert 71790738 n lWorld0x14d9c00
             idx  18 pdig b5f9e1ae53ecdd2dd902cfc7a5a6bf2c ndig      1 nprog 290266 nvert 71790322 n lOuterWaterPool0x14dbd60
             idx  19 pdig 6899ebf7e701f7a850951c59547eb53c ndig      1 nprog 290267 nvert 71790372 n lPoolLining0x14db8b0
             idx  20 pdig 68cfdc5d3977ec2d38c1e74b71ad3d2e ndig      1 nprog 290268 nvert 71790422 n lBtmRock0x14db220
             idx  21 pdig c81e728d9d4c2f636f067f89cc14862c ndig      1 nprog      0 nvert     96 n lUpperChimneyTyvek0x2547c80
             idx  22 pdig c4ca4238a0b923820dcc509a6f75849b ndig      1 nprog      0 nvert     96 n lUpperChimneySteel0x2547bb0
             idx  23 pdig cfcd208495d565ef66e7dff9f98764da ndig      1 nprog      0 nvert     50 n lUpperChimneyLS0x2547ae0
             idx  24 pdig 746ea22d5a1acd3c4f37fe7d648e9767 ndig      1 nprog      3 nvert    292 n lUpperChimney0x2547a50
             idx  25 pdig a80eb4230c6c09bc0536ad95784c6f78 ndig      1 nprog      4 nvert    300 n lExpHall0x14da8d0
             idx  26 pdig 6d794eba2efcefa04607dadd5443354a ndig      1 nprog      5 nvert    308 n lTopRock0x14da5a0
             idx  27 pdig 1e6ef35822ab41ec6c862b0ee8686afa ndig      1 nprog      5 nvert   1602 n lLowerChimney0x254aa20
             idx  28 pdig 3c59dc048e8850243be8079a5c74d079 ndig      1 nprog      0 nvert    430 n lLowerChimneyTyvek0x254ab60
             idx  29 pdig b6d767d2f8ed5d21a44b0e5886680cb9 ndig      1 nprog      0 nvert     96 n lLowerChimneyAcrylic0x254ac30
             idx  30 pdig 37693cfc748049e45d87b8c7d8b9aacd ndig      1 nprog      0 nvert    700 n lLowerChimneySteel0x254ad00
             idx  31 pdig a4410129b0d223b22cbfe1aa3bb5f469 ndig      1 nprog      1 nvert    274 n lLowerChimneyLS0x254ad90
             idx  32 pdig 1ff1de774005f8da13f42943881c655f ndig      1 nprog      0 nvert    168 n lLowerChimneyBlocker0x254ae60
             idx  33 pdig 7ff5c8a0d84d0f85355630af10cf9176 ndig      1 nprog      1 nvert   1204 n lSurftube0x254b8d0
             idx  34 pdig 02e74f10e0327ad868d138f2b4fdd6f0 ndig      1 nprog      0 nvert    602 n lvacSurftube0x254ba90

    2017-08-17 14:05:30.178 INFO  [213429] [GTreeCheck::dumpRepeatCandidates@305] GTreeCheck::dumpRepeatCandidates 
     pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 placements  36572 n PMT_3inch_log0x1c9ef80
     pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 placements  17739 n lMaskVirtual0x1816910
     pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 placements    480 n lFasteners0x1506370
     pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 placements    480 n lSteel0x14dde40

    2017-08-17 14:05:30.538 INFO  [213429] [GTreeCheck::labelTree@377] GTreeCheck::labelTree count of non-zero setRepeatIndex 290254
    2017-08-17 14:05:43.338 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 0 numPlacements 1 numSolids 290276
    2017-08-17 14:05:43.596 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 1 numPlacements 36572 numSolids 5
    2017-08-17 14:05:43.809 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 2 numPlacements 17739 numSolids 6
    2017-08-17 14:05:44.019 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 3 numPlacements 480 numSolids 1
    2017-08-17 14:05:44.229 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 4 numPlacements 480 numSolids 1



Similar in NScene
-------------------

::

    2017-08-17 17:18:38.741 INFO  [276315] [NScene::init@196] NScene::init import_r START 
    2017-08-17 17:18:47.493 INFO  [276315] [NScene::init@200] NScene::init import_r DONE 
    2017-08-17 17:18:47.494 INFO  [276315] [NScene::init@204] NScene::init triple_debug  num_gltf_nodes 290276 triple_mismatch 10932
    2017-08-17 17:18:47.665 INFO  [276315] [NScene::postimportnd@616] NScene::postimportnd numNd 290276 num_selected 290276 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-08-17 17:18:51.272 INFO  [276315] [NScene::count_progeny_digests@990] NScene::count_progeny_digests verbosity 1 node_count 290276 digest_size 35
     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 
     **  ##  idx   0 pdig 68a31892bccd1741cc098d232c702605 num_pdig  36572 num_progeny      4 NScene::meshmeta mesh_id  22 lvidx  20 height  1 soname        PMT_3inch_pmt_solid0x1c9e270 lvname              PMT_3inch_log0x1c9ef80
     **      idx   1 pdig 683529bb1b0fedc340f2ebce47468395 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  26 lvidx  19 height  0 soname       PMT_3inch_cntr_solid0x1c9e640 lvname         PMT_3inch_cntr_log0x1c9f1f0
     **      idx   2 pdig c81fb13777b701cb8ce6cdb7f0661f1b num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  25 lvidx  17 height  0 soname PMT_3inch_inner2_solid_ell_helper0x1c9e5d0 lvname       PMT_3inch_inner2_log0x1c9f120
     **      idx   3 pdig 83a5a282f092aa7baf6982b54227bb54 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  24 lvidx  16 height  0 soname PMT_3inch_inner1_solid_ell_helper0x1c9e510 lvname       PMT_3inch_inner1_log0x1c9f050
     **      idx   4 pdig 50308873a9847d1c2c2029b6c9de7eeb num_pdig  36572 num_progeny      2 NScene::meshmeta mesh_id  23 lvidx  18 height  0 soname PMT_3inch_body_solid_ell_ell_helper0x1c9e4a0 lvname         PMT_3inch_body_log0x1c9eef0

     **      idx   5 pdig 27a989a1aeab2b96cedd2b6c4a7cba2f num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  17 lvidx  10 height  2 soname                      sMask0x1816f50 lvname                      lMask0x18170e0
     **      idx   6 pdig e39a411b54c3ce46fd382fef7f632157 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  21 lvidx  12 height  4 soname    PMT_20inch_inner2_solid0x1863010 lvname      PMT_20inch_inner2_log0x1863310
     **      idx   7 pdig 74d8ce91d143cad52fad9d3661dded18 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  20 lvidx  11 height  4 soname    PMT_20inch_inner1_solid0x1814a90 lvname      PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig a80803364fbf92f1b083ebff420b6134 num_pdig  17739 num_progeny      2 NScene::meshmeta mesh_id  19 lvidx  13 height  3 soname      PMT_20inch_body_solid0x1813ec0 lvname        PMT_20inch_body_log0x1863160
     **      idx   9 pdig 6b1283d04ffc8a27e19f84e2bec2ddd6 num_pdig  17739 num_progeny      3 NScene::meshmeta mesh_id  18 lvidx  14 height  3 soname       PMT_20inch_pmt_solid0x1813600 lvname             PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig 8cbe68d7d5c763820ff67b8088e0de98 num_pdig  17739 num_progeny      5 NScene::meshmeta mesh_id  16 lvidx  15 height  0 soname              sMask_virtual0x18163c0 lvname               lMaskVirtual0x1816910

     **  ##  idx  11 pdig ad8b68a55505a09ac7578f32418904b3 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  15 lvidx   9 height  2 soname                 sFasteners0x1506180 lvname                 lFasteners0x1506370
     **  ##  idx  12 pdig f93b8bbbac89ea22bac0bf188ba49a61 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  14 lvidx   8 height  1 soname                     sStrut0x14ddd50 lvname                     lSteel0x14dde40

             idx  13 pdig 7e51746feafa7f2621f71943da8f603c num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  13 lvidx   6 height  1 soname                    sTarget0x14dd640 lvname                    lTarget0x14dd830
             idx  14 pdig c1cb7d90c1b21d9244fb041363a01416 num_pdig      1 num_progeny      1 NScene::meshmeta mesh_id  12 lvidx   7 height  1 soname                   sAcrylic0x14dd0a0 lvname                   lAcrylic0x14dd290
             idx  15 pdig 2a8e6c1bbc5183cd347725e7525758de num_pdig      1 num_progeny 290264 NScene::meshmeta mesh_id  11 lvidx  29 height  1 soname                sInnerWater0x14dcb00 lvname                lInnerWater0x14dccf0
             idx  16 pdig 9c629989608370c2cfcdd13000efd779 num_pdig      1 num_progeny 290265 NScene::meshmeta mesh_id  10 lvidx  30 height  1 soname             sReflectorInCD0x14dc560 lvname             lReflectorInCD0x14dc750
             idx  17 pdig d05b109737bc8db360f7c1d7c9e435ce num_pdig      1 num_progeny 290275 NScene::meshmeta mesh_id   0 lvidx  34 height  0 soname                     sWorld0x14d9850 lvname                     lWorld0x14d9c00
             idx  18 pdig 1401822f0db9e6eecdff1c2bf1ccfdc7 num_pdig      1 num_progeny 290266 NScene::meshmeta mesh_id   9 lvidx  31 height  0 soname            sOuterWaterPool0x14dbc70 lvname            lOuterWaterPool0x14dbd60
             idx  19 pdig 5b3b8c2e2e10f565302ca085917c5b6e num_pdig      1 num_progeny 290267 NScene::meshmeta mesh_id   8 lvidx  32 height  0 soname                sPoolLining0x14db2e0 lvname                lPoolLining0x14db8b0
             idx  20 pdig b0b2c346a748c9d728a3d8820ab0f4fa num_pdig      1 num_progeny 290268 NScene::meshmeta mesh_id   7 lvidx  33 height  0 soname                sBottomRock0x14dab90 lvname                   lBtmRock0x14db220
             idx  21 pdig 3d2f8900f2e49c02b481c2f717aa9020 num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id   6 lvidx   2 height  1 soname           Upper_Tyvek_tube0x2547990 lvname         lUpperChimneyTyvek0x2547c80
             idx  22 pdig 4e44f1ac85cd60e3caa56bfd4afb675e num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id   5 lvidx   1 height  1 soname           Upper_Steel_tube0x2547890 lvname         lUpperChimneySteel0x2547bb0
             idx  23 pdig 011ecee7d295c066ae68d4396215c3d0 num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id   4 lvidx   0 height  0 soname              Upper_LS_tube0x2547790 lvname            lUpperChimneyLS0x2547ae0
             idx  24 pdig 0b6f5322017121bc6a01b06429b96ce1 num_pdig      1 num_progeny      3 NScene::meshmeta mesh_id   3 lvidx   3 height  0 soname              Upper_Chimney0x25476d0 lvname              lUpperChimney0x2547a50
             idx  25 pdig 233607c26ba9bdb41341dd85c6e2d272 num_pdig      1 num_progeny      4 NScene::meshmeta mesh_id   2 lvidx   4 height  0 soname                   sExpHall0x14da850 lvname                   lExpHall0x14da8d0
             idx  26 pdig 7f1ea14cfc666324859d3ab689041406 num_pdig      1 num_progeny      5 NScene::meshmeta mesh_id   1 lvidx   5 height  0 soname                   sTopRock0x14da370 lvname                   lTopRock0x14da5a0
             idx  27 pdig 8ea531d2ec901e4d1bda3f1db96f6ff6 num_pdig      1 num_progeny      5 NScene::meshmeta mesh_id  27 lvidx  26 height  1 soname            upper_tubeTyvek0x254a890 lvname              lLowerChimney0x254aa20
             idx  28 pdig 29bdbc822df2e6c13dcf4afe6913525f num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  28 lvidx  21 height  3 soname                   unionLS10x2548db0 lvname         lLowerChimneyTyvek0x254ab60
             idx  29 pdig 70b48809e0305276c9defa82d51fb48c num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  29 lvidx  22 height  1 soname                AcrylicTube0x2548f40 lvname       lLowerChimneyAcrylic0x254ac30
             idx  30 pdig 4db87140662bd68076ef786f7163cedc num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  30 lvidx  23 height  4 soname                 unionSteel0x2549960 lvname         lLowerChimneySteel0x254ad00
             idx  31 pdig 6912d4b84d2d2e7f6cfd02bc50fe664b num_pdig      1 num_progeny      1 NScene::meshmeta mesh_id  31 lvidx  25 height  1 soname                   unionLS10x2549c00 lvname            lLowerChimneyLS0x254ad90
             idx  32 pdig 817808d063b210535f9a3ebbf173ea3d num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  32 lvidx  24 height  5 soname               unionBlocker0x254a570 lvname       lLowerChimneyBlocker0x254ae60
             idx  33 pdig e3f8899d3e08412c1a95878e3d4e9943 num_pdig      1 num_progeny      1 NScene::meshmeta mesh_id  33 lvidx  28 height  0 soname                  sSurftube0x2548170 lvname                  lSurftube0x254b8d0
             idx  34 pdig 5ff05a9d6ad1d0373d6cfaf43a9d1228 num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  34 lvidx  27 height  0 soname               svacSurftube0x254ba10 lvname               lvacSurftube0x254ba90
    2017-08-17 17:18:54.482 INFO  [276315] [NScene::postimportmesh@634] NScene::postimportmesh numNd 290276 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-08-17 17:18:54.482 INFO  [276315] [BConfig::dump@39] NScene::postimportmesh.cfg eki 13



Check viz 
------------

::

    op --j1707 --tracer --gltf 3 --debugger


