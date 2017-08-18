j1707 node accounting
========================

Issue : more than quarter million nodes
-----------------------------------------

* is that really the case ? YES
* are they maximally instanced ?  ALMOST, 480x fastener+strut COULD BE SIBLING INSTANCED
* get a feeling for tree structure


Sibling Instance Grouping ?
------------------------------

* grouping 3inch together with 20inch unfortunately impossible : they have complicated layouts
  that do not align to each other

* BUT the 480x fastener+strut looks like it could be instanced together 


To tabulate
--------------

* node counts
* instance counts, node counts per instance
* node counts in non-instanced globals
* buffer sizes, opengl(triangulated) optix (analytic+triangulated)


Analytic Buffer counts
------------------------

::

    2017-08-17 21:04:33.455 INFO  [348837] [OGeo::convert@203] OGeo::convert DONE  numMergedMesh: 5
    2017-08-17 21:04:33.455 INFO  [348837] [OGeo::dumpStats@572] OGeo::dumpStats num_stats 5
     mmIndex   0 numPrim    22 numPart   146 numTran(triples)    35 numPlan     0
     mmIndex   1 numPrim     5 numPart     7 numTran(triples)     5 numPlan     0
     mmIndex   2 numPrim     6 numPart   100 numTran(triples)    23 numPlan     0
     mmIndex   3 numPrim     1 numPart     7 numTran(triples)     2 numPlan     0
     mmIndex   4 numPrim     1 numPart     3 numTran(triples)     1 numPlan     0

* adding the complete binary tree node counts from the heights of each reproduces the numPart 



From geocache creation : note all 290276 volumes listed for mm0 : check implications on buffer sizes
-------------------------------------------------------------------------------------------------------

::

    op --j1707 -G

    210     2017-08-17 14:05:30.538 INFO  [213429] [GTreeCheck::labelTree@377] GTreeCheck::labelTree count of non-zero setRepeatIndex 290254
    211     2017-08-17 14:05:43.338 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 0 numPlacements 1 numSolids 290276
    212     2017-08-17 14:05:43.596 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 1 numPlacements 36572 numSolids 5
    213     2017-08-17 14:05:43.809 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 2 numPlacements 17739 numSolids 6
    214     2017-08-17 14:05:44.019 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 3 numPlacements 480 numSolids 1
    215     2017-08-17 14:05:44.229 INFO  [213429] [*GTreeCheck::makeAnalyticInstanceIdentityBuffer@517] GTreeCheck::makeAnalyticInstanceIdentityBuffer  ridx 4 numPlacements 480 numSolids 1
    216

    36572*5 + 17739*6 + 480*1 + 480*1 = 290254

    290276 - 290254 = 22     ## 22 global volumes 

    35 - 5 - 6 - 1 - 1 = 22   ## subtract 4 instances solid counts from total number of distinct solids (35)   gives the remainder


NScene first/last mesh off-by-1 ? Where are the 22 ridx=0 global volumes ?
------------------------------------------------------------------------------

::

    2017-08-17 17:39:00.235 INFO  [282274] [NScene::labelTree@1391] NScene::labelTree label_count (non-zero ridx labelTree_r) 290254 num_repeat_candidates 4
    2017-08-17 17:39:00.235 INFO  [282274] [NScene::dumpRepeatCount@1429] NScene::dumpRepeatCount m_verbosity 1
     ridx   1 count 182860
     ridx   2 count 106434
     ridx   3 count   480
     ridx   4 count   480
    2017-08-17 17:39:00.236 INFO  [282274] [NScene::dumpRepeatCount@1446] NScene::dumpRepeatCount totCount 290254


::

    36572*5 = 182860
    17739*6 = 106434
    480*1   =    480
    480*1   =    480
             --------
              290254 



Calc total parts for each mm by adding complete binary tree counts
-------------------------------------------------------------------

* nodes in binary tree of height h is 2^(h+1) - 1

::

    In [5]: 2**(1+np.arange(0,10))-1
    Out[5]: array([   1,    3,    7,   15,   31,   63,  127,  255,  511, 1023])
                      0     1     2     3     4     5     6     7     8     9


::

    // :set nowrap

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

     heights 1,0,0,0,0 -> nodes 3+1+1+1+1 = 7        
                 

     **      idx   5 pdig 27a989a1aeab2b96cedd2b6c4a7cba2f num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  17 lvidx  10 height  2 soname                      sMask0x1816f50 lvname                      lMask0x18170e0
     **      idx   6 pdig e39a411b54c3ce46fd382fef7f632157 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  21 lvidx  12 height  4 soname    PMT_20inch_inner2_solid0x1863010 lvname      PMT_20inch_inner2_log0x1863310
     **      idx   7 pdig 74d8ce91d143cad52fad9d3661dded18 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  20 lvidx  11 height  4 soname    PMT_20inch_inner1_solid0x1814a90 lvname      PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig a80803364fbf92f1b083ebff420b6134 num_pdig  17739 num_progeny      2 NScene::meshmeta mesh_id  19 lvidx  13 height  3 soname      PMT_20inch_body_solid0x1813ec0 lvname        PMT_20inch_body_log0x1863160
     **      idx   9 pdig 6b1283d04ffc8a27e19f84e2bec2ddd6 num_pdig  17739 num_progeny      3 NScene::meshmeta mesh_id  18 lvidx  14 height  3 soname       PMT_20inch_pmt_solid0x1813600 lvname             PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig 8cbe68d7d5c763820ff67b8088e0de98 num_pdig  17739 num_progeny      5 NScene::meshmeta mesh_id  16 lvidx  15 height  0 soname              sMask_virtual0x18163c0 lvname               lMaskVirtual0x1816910

     heights 2,4,4,3,3,0 -> nodes 7+31+31+15+15+1 = 100 

     **  ##  idx  11 pdig ad8b68a55505a09ac7578f32418904b3 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  15 lvidx   9 height  2 soname                 sFasteners0x1506180 lvname                 lFasteners0x1506370

     height 2 -> nodes 7

     **  ##  idx  12 pdig f93b8bbbac89ea22bac0bf188ba49a61 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  14 lvidx   8 height  1 soname                     sStrut0x14ddd50 lvname                     lSteel0x14dde40

     height 1 -> nodes 3

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

    heights  1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,3,1,4,1,5,0,0 
             3+3+3+3+1+1+1+1+3+3+1+1+1+1+3+15+3+31+3+63+1+1 = 146






Eyeballing source GDML
----------------------------------

::

    rg () 
    { 
        vim -R /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml
    }

    simon:issues blyth$ wc -l /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml
      277195 /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml


4 repeaters are apparent::

    lSteel0x14dde40         ~250
    lFasteners0x1506370     ~400
    lMaskVirtual0x1816910   ~18k 
    PMT_3inch_log0x1c9ef80  ~36k  


After fix to catch lSteel GTreeCheck agrees::

    2017-08-17 14:05:30.178 INFO  [213429] [GTreeCheck::dumpRepeatCandidates@305] GTreeCheck::dumpRepeatCandidates 
     pdig 286d87035b7a25bf19d347835138861e ndig  36572 nprog      4 placements  36572 n PMT_3inch_log0x1c9ef80
     pdig bae5cf200f4756b124f4c0563d9e12b1 ndig  17739 nprog      5 placements  17739 n lMaskVirtual0x1816910
     pdig 45c48cce2e2d7fbdea1afc51c7c6ad26 ndig    480 nprog      0 placements    480 n lFasteners0x1506370
     pdig c9f0f895fb98ab9159f51fd0297e236d ndig    480 nprog      0 placements    480 n lSteel0x14dde40


::

    000051   <materials>
       241   </materials>

       243   <solids>
       ...
       551   </solids> 

       553   <structure>


       737     <volume name="lInnerWater0x14dccf0">
       738       <materialref ref="Water0x14d1d00"/>
       739       <solidref ref="sInnerWater0x14dcb00"/>
       740       <physvol name="pAcylic0x14dda00">
       741         <volumeref ref="lAcrylic0x14dd290"/>
       742       </physvol>

       743       <physvol name="lSteel_phys0x14e01d0">     ~250 
       744         <volumeref ref="lSteel0x14dde40"/>
       745         <position name="lSteel_phys0x14e01d0_pos" unit="mm" x="3871.31568302668" y="0" z="18213.1083256635"/>
       746         <rotation name="lSteel_phys0x14e01d0_rot" unit="deg" x="180" y="12" z="180"/>
       747       </physvol>
       ...
      3138       <physvol name="lSteel_phys0x1504b20">
      3139         <volumeref ref="lSteel0x14dde40"/>
      3140         <position name="lSteel_phys0x1504b20_pos" unit="mm" x="3739.40379995337" y="-1001.97022837138" z="-18213.1083256635"/>
      3141         <rotation name="lSteel_phys0x1504b20_rot" unit="deg" x="3.1488779024914" y="11.5853397932875" z="15.3195239528622"/>
      3142       </physvol>

      3143       <physvol name="lFasteners_phys0x15072a0">   ~2000/5 ~400
      3144         <volumeref ref="lFasteners0x1506370"/>
      3145         <position name="lFasteners_phys0x15072a0_pos" unit="mm" x="3706.23380051738" y="0" z="17436.4591306808"/>
      3146         <rotation name="lFasteners_phys0x15072a0_rot" unit="deg" x="180" y="12" z="180"/>
      3147       </physvol>
      ....
      5538       <physvol name="lFasteners_phys0x152f3a0">
      5539         <volumeref ref="lFasteners0x1506370"/>
      5540         <position name="lFasteners_phys0x152f3a0_pos" unit="mm" x="3579.94694618522" y="-959.243893176594" z="-17436.4591306808"/>
      5541         <rotation name="lFasteners_phys0x152f3a0_rot" unit="deg" x="3.1488779024914" y="11.5853397932875" z="15.3195239528622"/>
      5542       </physvol>

      5543       <physvol name="lMaskVirtual_phys0x1868ad0">    ~90000/5 ~18k
      5544         <volumeref ref="lMaskVirtual0x1816910"/>
      5545         <position name="lMaskVirtual_phys0x1868ad0_pos" unit="mm" x="1065.41160578968" y="0" z="19470.8730700564"/>
      5546         <rotation name="lMaskVirtual_phys0x1868ad0_rot" unit="deg" x="180" y="3.132" z="180"/>
      5547       </physvol>
     .....
     94233       <physvol name="lMaskVirtual_phys0x1c9d5f0">
     94234         <volumeref ref="lMaskVirtual0x1816910"/>
     94235         <position name="lMaskVirtual_phys0x1c9d5f0_pos" unit="mm" x="19495.6188393558" y="-271.023178062762" z="-312.07772670818"/>
     94236         <rotation name="lMaskVirtual_phys0x1c9d5f0_rot" unit="deg" x="40.9726060827552" y="88.785428615014" z="40.9789798622846"/>
     94237       </physvol>

     94238       <physvol name="PMT_3inch_log_phys0x181f1b0">   ~ (277097-94238)/5 ~36k 
     94239         <volumeref ref="PMT_3inch_log0x1c9ef80"/>
     94240         <position name="PMT_3inch_log_phys0x181f1b0_pos" unit="mm" x="1402.8418375672" y="247.35886562974" z="19397.7665820157"/>
     94241         <rotation name="PMT_3inch_log_phys0x181f1b0_rot" unit="deg" x="-179.269408113041" y="4.13608063277865" z="-169.973618119703"/>
     94242       </physvol>
      ....
    277093       <physvol name="PMT_3inch_log_phys0x2547230">
    277094         <volumeref ref="PMT_3inch_log0x1c9ef80"/>
    277095         <position name="PMT_3inch_log_phys0x2547230_pos" unit="mm" x="529.088922853645" y="-305.469632034802" z="-19440.4025991135"/>
    277096         <rotation name="PMT_3inch_log_phys0x2547230_rot" unit="deg" x="0.900222120901556" y="1.55878160365665" z="30.0122466708415"/>
    277097       </physvol>



    277135     <volume name="lWorld0x14d9c00">
    277136       <materialref ref="Galactic0x1476410"/>
    277137       <solidref ref="sWorld0x14d9850"/>
    277138       <physvol name="pTopRock0x14da630">
    277139         <volumeref ref="lTopRock0x14da5a0"/>
    277140         <position name="pTopRock0x14da630_pos" unit="mm" x="0" y="0" z="32550"/>

    ::
        
        In [3]: a = np.load(os.path.expandvars("$TMP/NScene_triple.npy"))  

        In [4]: a.shape
        Out[4]: (290276, 3, 4, 4)

        In [10]: a[1]   ## so these are in traversal order from World
        Out[10]: 
        array([[[     1.,      0.,      0.,      0.],
                [     0.,      1.,      0.,      0.],
                [     0.,      0.,      1.,      0.],
                [     0.,      0.,  32550.,      1.]],

               [[     1.,      0.,      0.,      0.],
                [     0.,      1.,      0.,      0.],
                [     0.,      0.,      1.,      0.],
                [     0.,      0., -32550.,      1.]],

               [[     1.,      0.,      0.,      0.],
                [     0.,      1.,      0.,      0.],
                [     0.,      0.,      1., -32550.],
                [     0.,      0.,      0.,      1.]]], dtype=float32)


    277141       </physvol>
    277142       <physvol name="pBtmRock0x14db9f0">
    277143         <volumeref ref="lBtmRock0x14db220"/>
    277144       </physvol>
    277145     </volume>
    277146     <skinsurface name="Tube_surf" surfaceproperty="TubeSurface">
    ......
    277185     <bordersurface name="CDTyvekSurface" surfaceproperty="CDTyvekOpticalSurface">
    277186       <physvolref ref="pOuterWaterPool0x14dba40"/>
    277187       <physvolref ref="pCentralDetector0x14ddb50"/>
    277188     </bordersurface>
    277189   </structure>
    277190 
    277191   <setup name="Default" version="1.0">
    277192     <world ref="lWorld0x14d9c00"/>
    277193   </setup>
    277194 
    277195 </gdml>
     



