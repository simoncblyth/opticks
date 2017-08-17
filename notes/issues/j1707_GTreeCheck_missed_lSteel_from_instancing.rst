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



Check viz 
------------

::

    op --j1707 --tracer --gltf 3 --debugger


