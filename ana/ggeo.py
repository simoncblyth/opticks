#!/usr/bin/env python
"""
ggeo.py
=========

See also GNodeLib.py 

TODO:

* connection between solids and PV names 






Dumping using single node index and triplet RPO (ridx/pidx/oidx repeat/placement/offset) indexing::

     ggeo.py 0         # world volume 
     ggeo.py 1/0/0     # first placement of outer volume of first repeat   
     ggeo.py 1/        # missing elements of the triplet default to 0 

     ggeo.py 2/0/0     # first placement of outer volume of second repeat     ridx/pidx/oidx

When using the triplet form of node specification wildcards are 
accepted in the first ridx slot (eg 5:9) and third oidx slot (eg *), 
for example dumping all all volumes in ridx 5,6,7,8 with::

    epsilon:issues blyth$ ggeo.py 5:9/0/* --brief 
    nidx: 69668 triplet: 5000000 sh:5f0014 sidx:    0   nrpo( 69668     5     0     0 )  shape(  95  20                       base_steel0x360d8f0                            Water///Steel) 
    nidx: 69078 triplet: 6000000 sh:5e0014 sidx:    0   nrpo( 69078     6     0     0 )  shape(  94  20                             uni10x34cdcb0                            Water///Steel) 
    nidx: 68488 triplet: 7000000 sh:5d0014 sidx:    0   nrpo( 68488     7     0     0 )  shape(  93  20                   sStrutBallhead0x352a360                            Water///Steel) 
    nidx: 70258 triplet: 8000000 sh:600010 sidx:    0   nrpo( 70258     8     0     0 )  shape(  96  16                     uni_acrylic30x35ff3d0                          Water///Acrylic) 


The --names option dumps PV and LV names, for example dumping PV, LV names of the first instance placement 
of all volumes in ridx 1 thru 4::

    epsilon:GItemList blyth$ ggeo.py 1:5/ --names
    nrpo( 176632     1     0     0 )                        PMT_3inch_log_phys0x4437d00                             PMT_3inch_log0x4436df0  114 PMT_3inch_pmt_solid0x4436210 
    nrpo( 176633     1     0     1 )                       PMT_3inch_body_phys0x4437230                        PMT_3inch_body_log0x4436ce0  112 PMT_3inch_body_solid_ell_ell_helper0x44364d0 
    nrpo( 176634     1     0     2 )                     PMT_3inch_inner1_phys0x44372b0                      PMT_3inch_inner1_log0x4436f00  110 PMT_3inch_inner1_solid_ell_helper0x4436560 
    nrpo( 176635     1     0     3 )                     PMT_3inch_inner2_phys0x4437360                      PMT_3inch_inner2_log0x4437010  111 PMT_3inch_inner2_solid_ell_helper0x4436640 
    nrpo( 176636     1     0     4 )                       PMT_3inch_cntr_phys0x4437410                        PMT_3inch_cntr_log0x4437120  113 PMT_3inch_cntr_solid0x44366d0 
    nrpo(  70960     2     0     0 )                         pLPMT_NNVT_MCPPMT0x3cbba60                    NNVTMCPPMTlMaskVirtual0x3cb41a0  103 NNVTMCPPMTsMask_virtual0x3cb3b40 
    nrpo(  70961     2     0     1 )                           NNVTMCPPMTpMask0x3c9fe00                           NNVTMCPPMTlMask0x3c9fc80   98 NNVTMCPPMTsMask0x3c9fa80 
    nrpo(  70962     2     0     2 )            NNVTMCPPMT_PMT_20inch_log_phys0x3c9fe80                 NNVTMCPPMT_PMT_20inch_log0x3caec40  102 NNVTMCPPMT_PMT_20inch_pmt_solid0x3ca9320 
    nrpo(  70963     2     0     3 )           NNVTMCPPMT_PMT_20inch_body_phys0x3caefa0            NNVTMCPPMT_PMT_20inch_body_log0x3caeb60  101 NNVTMCPPMT_PMT_20inch_body_solid0x3cad240 
    nrpo(  70964     2     0     4 )         NNVTMCPPMT_PMT_20inch_inner1_phys0x3caf030          NNVTMCPPMT_PMT_20inch_inner1_log0x3caed60   99 NNVTMCPPMT_PMT_20inch_inner1_solid_1_Ellipsoid0x3503950 
    nrpo(  70965     2     0     5 )         NNVTMCPPMT_PMT_20inch_inner2_phys0x3caf0f0          NNVTMCPPMT_PMT_20inch_inner2_log0x3caee80  100 NNVTMCPPMT_PMT_20inch_inner2_solid0x3cae8f0 
    nrpo(  70966     3     0     0 )                    pLPMT_Hamamatsu_R128600x3cbbae0               HamamatsuR12860lMaskVirtual0x3c9a5c0  109 HamamatsuR12860sMask_virtual0x3c99fb0 
    nrpo(  70967     3     0     1 )                      HamamatsuR12860pMask0x3c9b320                      HamamatsuR12860lMask0x3c9b1a0  104 HamamatsuR12860sMask0x3c9afa0 
    nrpo(  70968     3     0     2 )       HamamatsuR12860_PMT_20inch_log_phys0x3c9b3b0            HamamatsuR12860_PMT_20inch_log0x3c93920  108 HamamatsuR12860_PMT_20inch_pmt_solid_1_90x3cb68e0 
    nrpo(  70969     3     0     3 )      HamamatsuR12860_PMT_20inch_body_phys0x345b3c0       HamamatsuR12860_PMT_20inch_body_log0x3c93830  107 HamamatsuR12860_PMT_20inch_body_solid_1_90x3ca7680 
    nrpo(  70970     3     0     4 )    HamamatsuR12860_PMT_20inch_inner1_phys0x3c94040     HamamatsuR12860_PMT_20inch_inner1_log0x345b160  105 HamamatsuR12860_PMT_20inch_inner1_solid_I0x3c96fa0 
    nrpo(  70971     3     0     5 )    HamamatsuR12860_PMT_20inch_inner2_phys0x3c94100     HamamatsuR12860_PMT_20inch_inner2_log0x345b290  106 HamamatsuR12860_PMT_20inch_inner2_solid_1_90x3c93610 
    nrpo( 304636     4     0     0 )     mask_PMT_20inch_vetolMaskVirtual_phys0x4433460          mask_PMT_20inch_vetolMaskVirtual0x3ca10e0  126 mask_PMT_20inch_vetosMask_virtual0x3ca0a80 
    nrpo( 304637     4     0     1 )                 mask_PMT_20inch_vetopMask0x3ca1e40                 mask_PMT_20inch_vetolMask0x3ca1cb0  121 mask_PMT_20inch_vetosMask0x3ca1aa0 
    nrpo( 304638     4     0     2 )                  PMT_20inch_veto_log_phys0x3ca5fa0                       PMT_20inch_veto_log0x3ca5470  125 PMT_20inch_veto_pmt_solid_1_20x3ca38b0 
    nrpo( 304639     4     0     3 )                 PMT_20inch_veto_body_phys0x3ca57a0                  PMT_20inch_veto_body_log0x3ca5360  124 PMT_20inch_veto_body_solid_1_20x3ca4230 
    nrpo( 304640     4     0     4 )               PMT_20inch_veto_inner1_phys0x3ca5820                PMT_20inch_veto_inner1_log0x3ca5580  122 PMT_20inch_veto_inner1_solid0x3ca4f10 
    nrpo( 304641     4     0     5 )               PMT_20inch_veto_inner2_phys0x3ca58d0                PMT_20inch_veto_inner2_log0x3ca5690  123 PMT_20inch_veto_inner2_solid0x3ca5130 

Same for ridx 5 thru 8::

    epsilon:GItemList blyth$ ggeo.py 5:9/ --names 
    nrpo(  69668     5     0     0 )                               lUpper_phys0x35b5ac0                                    lUpper0x35b5a00   95 base_steel0x360d8f0 
    nrpo(  69078     6     0     0 )                           lFasteners_phys0x34ce040                                lFasteners0x34cdf00   94 uni10x34cdcb0 
    nrpo(  68488     7     0     0 )                               lSteel_phys0x352c890                                    lSteel0x352c760   93 sStrutBallhead0x352a360 
    nrpo(  70258     8     0     0 )                            lAddition_phys0x35ff770                                 lAddition0x35ff5f0   96 uni_acrylic30x35ff3d0 

Names for first two placements of ridx 5 thru 8::

    epsilon:GItemList blyth$ ggeo.py 5:9/0:2 --names 
    nrpo(  69668     5     0     0 )                               lUpper_phys0x35b5ac0                                    lUpper0x35b5a00   95 base_steel0x360d8f0 
    nrpo(  69669     5     1     0 )                               lUpper_phys0x35b5bb0                                    lUpper0x35b5a00   95 base_steel0x360d8f0 
    nrpo(  69078     6     0     0 )                           lFasteners_phys0x34ce040                                lFasteners0x34cdf00   94 uni10x34cdcb0 
    nrpo(  69079     6     1     0 )                           lFasteners_phys0x34ce140                                lFasteners0x34cdf00   94 uni10x34cdcb0 
    nrpo(  68488     7     0     0 )                               lSteel_phys0x352c890                                    lSteel0x352c760   93 sStrutBallhead0x352a360 
    nrpo(  68489     7     1     0 )                               lSteel_phys0x352a4a0                                    lSteel0x352c760   93 sStrutBallhead0x352a360 
    nrpo(  70258     8     0     0 )                            lAddition_phys0x35ff770                                 lAddition0x35ff5f0   96 uni_acrylic30x35ff3d0 
    nrpo(  70259     8     1     0 )                            lAddition_phys0x35ff870                                 lAddition0x35ff5f0   96 uni_acrylic30x35ff3d0 


Using suppression of some prolific names can dump all interesting names at once:: 

    epsilon:ana blyth$ ggeo.sh 0:10/ --names --suppress
    python3 /Users/blyth/opticks/ana/ggeo.py 0:10/ --names --suppress
    [2021-04-24 13:57:42,444] p41483 {/Users/blyth/opticks/ana/ggeo.py:777} INFO - using suppression (HBeam|ixture|anchor|Steel2|Plane|Wall|Receiver|Strut0x|sBar0x) specificed by envvar OPTICKS_GGEO_SUPPRESS 
      nrpo(      0     0     0     0 )                                 lWorld0x33e33d0_PV                                    lWorld0x33e33d0  130 sWorld0x33e3370 
      nrpo(      1     0     0     1 )                                  pTopRock0x33f3c00                                  lTopRock0x33f3b30   12 sTopRock0x33f3aa0 
      nrpo(      2     0     0     2 )                                  pExpHall0x33f40b0                                  lExpHall0x33f3fb0   11 sExpHall0x33f3f20 
      nrpo(      3     0     0     3 )                        lUpperChimney_phys0x4e6e210                             lUpperChimney0x4e6c7a0    3 Upper_Chimney0x4e6c340 
      nrpo(      4     0     0     4 )                           pUpperChimneyLS0x4e6cbc0                           lUpperChimneyLS0x4e6c8a0    0 Upper_LS_tube0x4e6c450 
      nrpo(      5     0     0     5 )                        pUpperChimneySteel0x4e6cc90                        lUpperChimneySteel0x4e6c9b0    1 Upper_Steel_tube0x4e6c570 
      nrpo(      6     0     0     6 )                        pUpperChimneyTyvek0x4e6cd60                        lUpperChimneyTyvek0x4e6cac0    2 Upper_Tyvek_tube0x4e6c690 
      nrpo(      7     0     0     7 )                               pTopTracker0x4e7f260                                    lAirTT0x4e713c0   10 sAirTT0x4e71260 
      nrpo(  65717     0     0   197 )                                  pBtmRock0x33f9200                                  lBtmRock0x33f89c0  129 sBottomRock0x33f4390 
      nrpo(  65718     0     0   198 )                               pPoolLining0x33f9160                               lPoolLining0x33f90a0  128 sPoolLining0x33f8a80 
      nrpo(  65719     0     0   199 )                           pOuterWaterPool0x3490fa0                           lOuterWaterPool0x33f9470  127 sOuterWaterPool0x33f9360 
      nrpo(  67840     0     0  2320 )                          pCentralDetector0x3492d70                            lReflectorInCD0x34916c0  120 sReflectorInCD0x3491470 
      nrpo(  67841     0     0  2321 )                               pInnerWater0x3492b80                               lInnerWater0x3491d30  119 sInnerWater0x3491ae0 
      nrpo(  67842     0     0  2322 )                                  pAcrylic0x3492c20                                  lAcrylic0x34923a0   90 sAcrylic0x3492150 
      nrpo(  67843     0     0  2323 )                                   pTarget0x3492cc0                                   lTarget0x3492a10   89 sTarget0x34927c0 
      nrpo( 304632     0     0  3080 )                        lLowerChimney_phys0x4e706b0                             lLowerChimney0x4e6eac0  118 sWaterTube0x4e6e9b0 
      nrpo( 304633     0     0  3081 )                      pLowerChimneyAcrylic0x4e6f220                      lLowerChimneyAcrylic0x4e6ece0  115 sChimneyAcrylic0x4e6ebc0 
      nrpo( 304634     0     0  3082 )                           pLowerChimneyLS0x4e6f2e0                           lLowerChimneyLS0x4e6eef0  116 sChimneyLS0x4e6ede0 
      nrpo( 304635     0     0  3083 )                        pLowerChimneySteel0x4e6f3b0                        lLowerChimneySteel0x4e6f110  117 sChimneySteel0x4e6eff0 
      nrpo( 176632     1     0     0 )                        PMT_3inch_log_phys0x43c2530                             PMT_3inch_log0x43c1620  114 PMT_3inch_pmt_solid0x43c0a40 
      nrpo( 176633     1     0     1 )                       PMT_3inch_body_phys0x43c1a60                        PMT_3inch_body_log0x43c1510  112 PMT_3inch_body_solid_ell_ell_helper0x43c0d00 
      nrpo( 176634     1     0     2 )                     PMT_3inch_inner1_phys0x43c1ae0                      PMT_3inch_inner1_log0x43c1730  110 PMT_3inch_inner1_solid_ell_helper0x43c0d90 
      nrpo( 176635     1     0     3 )                     PMT_3inch_inner2_phys0x43c1b90                      PMT_3inch_inner2_log0x43c1840  111 PMT_3inch_inner2_solid_ell_helper0x43c0e70 
      nrpo( 176636     1     0     4 )                       PMT_3inch_cntr_phys0x43c1c40                        PMT_3inch_cntr_log0x43c1950  113 PMT_3inch_cntr_solid0x43c0f00 
      nrpo(  70961     2     0     1 )                           NNVTMCPPMTpMask0x3c2cad0                           NNVTMCPPMTlMask0x3c2c950   98 NNVTMCPPMTsMask0x3c2c750 
      nrpo(  70962     2     0     2 )            NNVTMCPPMT_PMT_20inch_log_phys0x3c2cb50                 NNVTMCPPMT_PMT_20inch_log0x3c2a6b0  102 NNVTMCPPMT_PMT_20inch_pmt_solid0x3c21980 
      nrpo(  70963     2     0     3 )           NNVTMCPPMT_PMT_20inch_body_phys0x3c2aa10            NNVTMCPPMT_PMT_20inch_body_log0x3c2a5d0  101 NNVTMCPPMT_PMT_20inch_body_solid0x3c258a0 
      nrpo(  70964     2     0     4 )         NNVTMCPPMT_PMT_20inch_inner1_phys0x3c2aaa0          NNVTMCPPMT_PMT_20inch_inner1_log0x3c2a7d0   99 NNVTMCPPMT_PMT_20inch_inner1_solid_1_Ellipsoid0x3497520 
      nrpo(  70965     2     0     5 )         NNVTMCPPMT_PMT_20inch_inner2_phys0x3c2ab60          NNVTMCPPMT_PMT_20inch_inner2_log0x3c2a8f0  100 NNVTMCPPMT_PMT_20inch_inner2_solid0x3c2a360 
      nrpo(  70967     3     0     1 )                      HamamatsuR12860pMask0x3c394b0                      HamamatsuR12860lMask0x3c39330  104 HamamatsuR12860sMask0x3c39130 
      nrpo(  70968     3     0     2 )       HamamatsuR12860_PMT_20inch_log_phys0x3c39540            HamamatsuR12860_PMT_20inch_log0x3c36c90  108 HamamatsuR12860_PMT_20inch_pmt_solid_1_90x3c4a970 
      nrpo(  70969     3     0     3 )      HamamatsuR12860_PMT_20inch_body_phys0x33eeec0       HamamatsuR12860_PMT_20inch_body_log0x3c36ba0  107 HamamatsuR12860_PMT_20inch_body_solid_1_90x3c28080 
      nrpo(  70970     3     0     4 )    HamamatsuR12860_PMT_20inch_inner1_phys0x3c373b0     HamamatsuR12860_PMT_20inch_inner1_log0x33eec60  105 HamamatsuR12860_PMT_20inch_inner1_solid_I0x3c32bc0 
      nrpo(  70971     3     0     5 )    HamamatsuR12860_PMT_20inch_inner2_phys0x3c37470     HamamatsuR12860_PMT_20inch_inner2_log0x33eed90  106 HamamatsuR12860_PMT_20inch_inner2_solid_1_90x3c36980 
      nrpo( 304637     4     0     1 )                 mask_PMT_20inch_vetopMask0x3c2eb60                 mask_PMT_20inch_vetolMask0x3c2e9d0  121 mask_PMT_20inch_vetosMask0x3c2e7c0 
      nrpo( 304638     4     0     2 )                  PMT_20inch_veto_log_phys0x3c3e950                       PMT_20inch_veto_log0x3c3de20  125 PMT_20inch_veto_pmt_solid_1_20x3c305d0 
      nrpo( 304639     4     0     3 )                 PMT_20inch_veto_body_phys0x3c3e150                  PMT_20inch_veto_body_log0x3c3dd10  124 PMT_20inch_veto_body_solid_1_20x3c3cc50 
      nrpo( 304640     4     0     4 )               PMT_20inch_veto_inner1_phys0x3c3e1d0                PMT_20inch_veto_inner1_log0x3c3df30  122 PMT_20inch_veto_inner1_solid0x3c3d8c0 
      nrpo( 304641     4     0     5 )               PMT_20inch_veto_inner2_phys0x3c3e280                PMT_20inch_veto_inner2_log0x3c3e040  123 PMT_20inch_veto_inner2_solid0x3c3dae0 
      nrpo(  68488     5     0     0 )                               lSteel_phys0x34c07b0                                    lSteel0x34c0680   93 sStrutBallhead0x34be280 
      nrpo(  69078     6     0     0 )                           lFasteners_phys0x3461f60                                lFasteners0x3461e20   94 uni10x3461bd0 
      nrpo(  69668     7     0     0 )                               lUpper_phys0x35499e0                                    lUpper0x3549920   95 base_steel0x35a1810 
      nrpo(  70258     8     0     0 )                            lAddition_phys0x3593690                                 lAddition0x3593510   96 uni_acrylic30x35932f0 
      nrpo(     10     9     0     0 )                               pPanel_0_f_0x4e7c3c0                                    lPanel0x4e71970    7 sPanel0x4e71750 
      nrpo(     11     9     0     1 )                                pPanelTape0x4e7c6a0                                lPanelTape0x4e71b00    6 sPanelTape0x4e71a70 
    [2021-04-24 13:57:42,496] p41483 {/Users/blyth/opticks/ana/ggeo.py:601} INFO - supressed 3193 volumes 
    epsilon:ana blyth$ 






A convenient visualization workflow is to use the above python triple indexing to find PV names to target, eg::

    OTracerTest --targetpvn lFasteners_phys   ## do not include the 0x reference in the targetted name, as it will differ between machines/invokations 
    OTracerTest --target    69078             ## using raw indices is NOT advisable as they go stale very quickly with changed geometry

Volume idsmry dumping::

    epsilon:ana blyth$ ggeo.py 3199 -i
    iden(  3199    5000000     2f001b     -1 )  nrpo(  3199     5     0     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 
    iden(  3200    5000001     2e001c     -1 )  nrpo(  3200     5     0     1 )  shape(  46  28            pmt-hemi-vac0xc21e2480x3e85290                           Pyrex///Vacuum) 
    iden(  3201    5000002     2b001d     -1 )  nrpo(  3201     5     0     2 )  shape(  43  29        pmt-hemi-cathode0xc2f1ce80x3e842d0                        Vacuum///Bialkali) 
    iden(  3202    5000003     2c001e     -1 )  nrpo(  3202     5     0     3 )  shape(  44  30            pmt-hemi-bot0xc22a9580x3e844c0                    Vacuum///OpaqueVacuum) 
    iden(  3203    5000004     2d001e     -1 )  nrpo(  3203     5     0     4 )  shape(  45  30         pmt-hemi-dynode0xc346c500x3e84610                    Vacuum///OpaqueVacuum) 
    iden(  3204        57f     30001f     -1 )  nrpo(  3204     0     0  1407 )  shape(  48  31             AdPmtCollar0xc2c52600x3e86030          MineralOil///UnstStainlessSteel) 
    iden(  3205    5000100     2f001b     -1 )  nrpo(  3205     5     1     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 
    iden(  3206    5000101     2e001c     -1 )  nrpo(  3206     5     1     1 )  shape(  46  28            pmt-hemi-vac0xc21e2480x3e85290                           Pyrex///Vacuum) 
    iden(  3207    5000102     2b001d     -1 )  nrpo(  3207     5     1     2 )  shape(  43  29        pmt-hemi-cathode0xc2f1ce80x3e842d0                        Vacuum///Bialkali) 


::

    In [52]: gc(10)
    gc.identity[10]  nidx/midx/bidx/sidx  [530   8   3   1]  
    gc.mlibnames[10] : sPlane0x47c46c0 
    gc.blibnames[10] : Air///Air 

    gt : gc.transforms0[530]
    [[    0.       1.       0.       0.  ]
     [   -1.       0.       0.       0.  ]
     [    0.       0.       1.       0.  ]
     [20133.6  -6711.2  23504.15     1.  ]]

    tr : transform
    [[    0.       1.       0.       0.  ]
     [   -1.       0.       0.       0.  ]
     [    0.       0.       1.       0.  ]
     [20133.6  -6711.2  23504.15     1.  ]]

    it : inverted transform
    [[     0.       -1.        0.        0.  ]
     [     1.        0.        0.        0.  ]
     [    -0.        0.        1.        0.  ]
     [  6711.2   20133.6  -23504.15      1.  ]]

    bb : bbox4
    [[ 16748.4492 -10141.8008  23497.5         1.    ]
     [ 23518.75    -3280.6001  23510.8008      1.    ]]

    cbb : (bb[0]+bb[1])/2.
    [20133.5996 -6711.2004 23504.1504     1.    ]

    c4 : center4
    [20133.5996 -6711.2002 23504.1504     1.    ]

    ce : center_extent
    [20133.6    -6711.2    23504.15    3430.6003]

    ic4 : np.dot( c4, it) : inverse transform applied to center4 
    [0. 0. 0. 1.]

    ibb : np.dot( bb, it) : inverse transform applied to bbox4 
    [[-3430.6006  3385.1504    -6.6504     1.    ]
     [ 3430.6001 -3385.1504     6.6504     1.    ]]


"""
import os, re, sys, logging, argparse
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.blib import BLib
from opticks.ana.key import key_
from opticks.ana.OpticksIdentity import OpticksIdentity
from opticks.ana.rsttable import RSTTable

tx_load = lambda _:list(map(str.strip, open(_).readlines()))

def Three2Four(a, w=1): 
    """
    :param a: array shaped with last dimension 3 
    :param w: 1 or 0 (points or vectors)
    :return r: corresponding array which last dimension increased from 3 to 4
    """
    s = list(a.shape)
    assert s[-1] == 3, "unexpected shape %r , last dimension must be 3" % s
    assert w in (1,0), "w must be 1 or 0" 
    s[-1] = 4 
    b = np.ones(s) if w == 1 else np.zeros(s)
    d = len(s)
    if d == 1:
        b[:3] = a
    if d == 2:
        b[:,:3] = a
    elif d == 3:
        b[:,:,:3] = a
    elif d == 4:
        b[:,:,:,:3] = a
    else:
        assert 0, "unexpected shape %r " % s
    pass
    r = b
    return r 



class GGeo(object):
    KEY = key_(os.environ["OPTICKS_KEY"])
    KEYDIR = KEY.keydir
    VERSION = KEY.version
    SUPPRESS_ = os.environ.get("OPTICKS_GGEO_SUPPRESS", "") 
    SUPPRESS_PTN = "("+SUPPRESS_.replace(",","|")+")" 
    SUPPRESS = None if SUPPRESS_PTN == "()" else re.compile(SUPPRESS_PTN) 

    @classmethod
    def Suppress(cls, name):
        """
        :param name: 
        :return bool: True when the name contains one of the suppressed strings provided by envvar OPTICKS_GGEO_SUPPRESS
        """
        if cls.SUPPRESS is None:
            return False
        else:
            return not cls.SUPPRESS.search(name) is None 
        pass

    volume_names    = list(map(lambda _:"volume_%s" % _, "transforms center_extent bbox meshes nodeinfo identity".split()))
    placement_names = list(map(lambda _:"placement_%s"%_, "itransforms iidentity".split()))
    face_names      = list(map(lambda _:"face_%s"%_,  "sensors boundaries nodes indices".split()))
    vertex_names    = list(map(lambda _:"vertex_%s"%_,  "colors normals vertices".split()))
    names = volume_names + placement_names + face_names + vertex_names

    all_volume_names = list(map(lambda _:"all_volume_%s" % _, "nodeinfo identity center_extent bbox transforms inverse_transforms".split()))

    PV = "{keydir}/GNodeLib/all_volume_PVNames.txt"   # full node list of PV names
    LV = "{keydir}/GNodeLib/all_volume_LVNames.txt"   # full node list of LV names
    MS = "{keydir}/GItemList/GMeshLib.txt"            # list of unique solid names 
 
    @classmethod   
    def Path(cls, ridx, name, subdir="GMergedMesh", alldir="GNodeLib"): 
        """
        :param ridx: -1 for all volumes from GNodeLib, 0,1,2,3,... for GMergedMesh Composite "Solids"
        """
        keydir = cls.KEYDIR 
        if ridx == -1:
            fmt = "{keydir}/{alldir}/{name}.npy"
        elif ridx > -1: 
            fmt = "{keydir}/{subdir}/{ridx}/{name}.npy"
        else:
            assert 0
        pass 
        return os.path.expandvars(fmt.format(**locals()))

    @classmethod   
    def TxtPath(cls, name): 
        keydir = cls.KEYDIR 
        return os.path.expandvars("{keydir}/{name}".format(**locals()))

    @classmethod   
    def Array(cls, ridx, name): 
        path = cls.Path(ridx, name) 
        a = np.load(path)
        return a 

    @classmethod   
    def Txt(cls, name): 
        path = cls.TxtPath(name) 
        return np.array(tx_load(path)) 
        #return np.loadtxt(path, dtype="|S100")

    @classmethod   
    def Attn(cls, ridx, name): 
        return "_%s_%d" % (name, ridx) 

    def __init__(self, args=None):
        self.args = args
        self.suppress_count = 0 
        keydir = self.KEYDIR
        path = os.path.expandvars("{keydir}/GMergedMesh".format(**locals()))
        mmidx = sorted(map(int,os.listdir(path)))
        num_repeats = len(mmidx)
        self.num_repeats = num_repeats 
        blib = BLib(keydir)
        self.blib = np.array(blib.names().split("\n"))
        self.pv = np.loadtxt(self.PV.format(**locals()), dtype="|S100")
        self.lv = np.loadtxt(self.LV.format(**locals()), dtype="|S100")
        self.ms = np.loadtxt(self.MS.format(**locals()), dtype="|S100")
        self.msn = list(map(lambda _:_.decode('utf-8'),self.ms)) 

        self.mlib = self.get_txt("GItemList/GMeshLib.txt", "_mlib") 
        self.midx = (self.all_volume_identity[:,2] >> 16) & 0xffff  
        self.bidx = (self.all_volume_identity[:,2] >>  0) & 0xffff  
        self.mlibnames = self.mlib[self.midx]   # mesh/lv names
        self.blibnames = self.blib[self.bidx]   # boundary names
        self.mmidx = mmidx 

    #mlib = property(lambda self:self.get_txt("GItemList/GMeshLib.txt", "_mlib")) 
    #midx = property(lambda self:(self.all_volume_identity[:,2] >> 16) & 0xffff ) 
    #bidx = property(lambda self:(self.all_volume_identity[:,2] >>  0) & 0xffff ) 
    #mlibnames = property(lambda self:self.mlib[self.midx])   # mesh/lv names
    #blibnames = property(lambda self:self.blib[self.bidx])   # boundary names

    def get_array(self, ridx, name):
        """
        Array actually loaded only the first time
        """
        attn = self.Attn(ridx,name) 
        if getattr(self, attn, None) is None:
            a = self.Array(ridx, name)
            setattr(self, attn, a)
        pass
        return getattr(self, attn)

    def get_txt(self, name, attn):
        if getattr(self, attn, None) is None:
            a = self.Txt(name)
            setattr(self, attn, a)
        pass
        return getattr(self, attn)

    def summary0(self):
        log.info("num_repeats:{gg.num_repeats}".format(gg=self))
        for ridx in range(self.num_repeats):
            for name in self.names:
                a = self.get_array(ridx,name)
                print("{ridx:2d} {name:25s}  {shape!r}".format(ridx=ridx,name=name,shape=a.shape))
            pass
            print()
        pass 

    def get_tot_volumes(self):
        return self.get_num_volumes(-1)
    num_volumes = property(get_tot_volumes)

    def get_num_volumes(self, ridx):
        name = "all_volume_transforms" if ridx == -1 else "volume_transforms"
        a = self.get_array(ridx,name)
        return a.shape[0]

    def get_num_placements(self, ridx):
        if ridx == -1:
            return 1
        pass
        name = "placement_itransforms"
        a = self.get_array(ridx,name)
        return a.shape[0]

    def summary(self):
        """
        Array shapes
        """
        log.info("num_repeats:{gg.num_repeats}".format(gg=self))

        fmt = "{ridx:10d} {num_volumes:15d} {num_placements:15d} {num_placed_vol:15d}"
        print("%10s %15s %15s %15s" % ("ridx","num_volumes", "num_placements", "num_placed_vol"))
        tot_vol = 0 
        tot_vol_ = self.get_num_volumes(-1)
        for ridx in range(self.num_repeats):
            num_volumes = self.get_num_volumes(ridx)
            num_placements = self.get_num_placements(ridx)
            num_placed_vol = num_volumes*num_placements
            tot_vol += num_placed_vol
            print(fmt.format(**locals()))
        pass
        print("%10s %15s %15s %15d" % ("","","tot_vol:",tot_vol)) 
        print("%10s %15s %15s %15d" % ("","","tot_vol_:",tot_vol_)) 
        print()

        for name in self.names:
            shape = []
            for ridx in range(self.num_repeats):
                a = self.get_array(ridx,name)
                shape.append("{shape!r:20s}".format(shape=a.shape))
            pass     
            print("{name:25s} {shape}".format(name=name,shape="".join(shape)))
        pass
        for name in self.all_volume_names:
            ridx = -1
            a = self.get_array(ridx,name)
            print("{name:25s} {shape!r}".format(name=name,shape=a.shape))
        pass

    def get_all_transforms(self):
        """
        Access transforms of all volumes via triplet indexing.
        The ordering of the transforms is not the same as all_volume_transforms.
        However can still check the match using the identity info to find
        the node index.
        """
        log.info("get_all_transforms and do identity consistency check : triplet->node->triplet")
        tot_volumes = self.get_tot_volumes()
        tr = np.zeros([tot_volumes,4,4],dtype=np.float32)
        count = 0 
        for ridx in range(self.num_repeats):
            num_placements = self.get_num_placements(ridx)
            num_volumes = self.get_num_volumes(ridx)
            for pidx in range(num_placements):
                for oidx in range(num_volumes):
                    nidx = self.get_node_index(ridx,pidx,oidx)
                    tr[nidx] = self.get_transform(ridx,pidx,oidx)
                    count += 1 
                pass
            pass
        pass
        assert tot_volumes == count  
        all_volume_transforms = self.get_array(-1,"all_volume_transforms")
        assert np.allclose( all_volume_transforms, tr )
        return tr  


    def get_transform_n(self, nidx):
        all_volume_transforms = self.get_array(-1,"all_volume_transforms")
        return all_volume_transforms[nidx] 

    def get_inverse_transform_n(self, nidx):
        all_volume_inverse_transforms = self.get_array(-1,"all_volume_inverse_transforms")
        return all_volume_inverse_transforms[nidx] 


    def get_inverse_transform(self, ridx, pidx, oidx):
        """
        No triplet way to do this yet, have to go via node index
        """
        nidx = self.get_node_index(ridx,pidx,oidx)
        return self.get_inverse_transform_n(nidx)

    def get_transform(self, ridx, pidx, oidx):
        """
        :param ridx: repeat idx, 0 for remainder
        :param pidx: placement index of the instance, 0 for remainder
        :param oidx: offset index, within the instance or among the remainder

        DONE in get_all_transforms, verified both routes match for all nodes including the remainders
        """
        ## native (triplet access)
        placement_itransforms = self.get_array(ridx, "placement_itransforms") # identity for remainder
        volume_transforms = self.get_array(ridx, "volume_transforms")   # within the instance or from root for remainder
        itr = placement_itransforms[pidx]
        vtr = volume_transforms[oidx].reshape(4,4)

        ggt = np.dot( vtr, itr )  

        ## cross reference to the node index
        nidx = self.get_node_index(ridx,pidx,oidx)

        ## nodeindex access 
        ntr = self.get_transform_n(nidx)

        assert np.allclose( ggt, ntr )
        return ggt

    def get_node_index(self, ridx, pidx, oidx):
        """
        :param ridx: repeat index, 0 for remainder
        :param pidx: placement index of the instance, 0 for remainder
        :param oidx: offset index, within the instance or among the remainder
        :return nidx: all_volume node index 

        The node index obtained from the placement_identity is used
        to do a reverse conversion check using nrpo, looking up 
        the triplet identity from the node index. These indices
        are consistency checked with the inputs.
        """
        placement_iidentity = self.get_array(ridx, "placement_iidentity")  # eg shape  (672, 5, 4)
        iid = placement_iidentity[pidx, oidx]
        nidx = iid[0]    

        nidx2,ridx2,pidx2,oidx2 = self.nrpo[nidx]
        assert nidx2 == nidx 
        assert ridx2 == ridx 
        assert pidx2 == pidx 

        if oidx2 != oidx:
            log.debug("mismatch oidx2:%d(from nrpo/iid)  oidx:%d(from range(num_volumes)) " % (oidx2, oidx)) 
        pass
        #assert oidx2 == oidx 
        return nidx 


    def get_lvidx(self, ridx, pidx, oidx):
        nidx = self.get_node_index(ridx,pidx,oidx)
        midx = self.midx[nidx] 
        return midx
   
    def get_soname(self, ridx, pidx, oidx):
        midx = self.get_lvidx(ridx, pidx, oidx)
        msn = self.msn[midx]
        return msn 

    def make_nrpo(self):
        """
        See okc/OpticksIdentity::Decode
        """
        gg = self
        avi = gg.all_volume_identity
        tid = avi[:,1] 
        nrpo = OpticksIdentity.NRPO(tid)
        return nrpo

    def _get_nrpo(self):
        if getattr(self,'_nrpo',None) is None:
            setattr(self,'_nrpo',self.make_nrpo())
        return self._nrpo 
    nrpo = property(_get_nrpo)

    def get_triplet_index(self, nidx):
        """
         cf ggeo/GGeo::getIdentity

        :param nidx: all_volume node index 
        :return nidx,ridx,pidx,oidx:
        """
        return self.nrpo[nidx]


    all_volume_center_extent = property(lambda self:self.get_array(-1,"all_volume_center_extent"))
    all_volume_bbox          = property(lambda self:self.get_array(-1,"all_volume_bbox"))
    all_volume_identity      = property(lambda self:self.get_array(-1,"all_volume_identity"))
    all_volume_transforms    = property(lambda self:self.get_array(-1,"all_volume_transforms"))  


    volume_transforms0  = property(lambda self:self.get_array(0,"volume_transforms")) 
    volume_transforms1  = property(lambda self:self.get_array(1,"volume_transforms"))
    volume_transforms2  = property(lambda self:self.get_array(2,"volume_transforms"))
    volume_transforms3  = property(lambda self:self.get_array(3,"volume_transforms"))
    volume_transforms4  = property(lambda self:self.get_array(4,"volume_transforms"))
    volume_transforms5  = property(lambda self:self.get_array(5,"volume_transforms"))

    placement_itransforms0 = property(lambda self:self.get_array(0,"placement_itransforms"))
    placement_itransforms1 = property(lambda self:self.get_array(1,"placement_itransforms"))
    placement_itransforms2 = property(lambda self:self.get_array(2,"placement_itransforms"))
    placement_itransforms3 = property(lambda self:self.get_array(3,"placement_itransforms"))
    placement_itransforms4 = property(lambda self:self.get_array(4,"placement_itransforms"))
    placement_itransforms5 = property(lambda self:self.get_array(5,"placement_itransforms"))

    placement_iidentity0 = property(lambda self:self.get_array(0,"placement_iidentity"))
    placement_iidentity1 = property(lambda self:self.get_array(1,"placement_iidentity"))
    placement_iidentity2 = property(lambda self:self.get_array(2,"placement_iidentity"))
    placement_iidentity3 = property(lambda self:self.get_array(3,"placement_iidentity"))
    placement_iidentity4 = property(lambda self:self.get_array(4,"placement_iidentity"))
    placement_iidentity5 = property(lambda self:self.get_array(5,"placement_iidentity"))


    def __call__(self,*args):
        """
        A single integer argument is interpreted as a node index (nidx), 
        otherwise 2 or 3 args are interpreted as ridx,pidx,oidx with 
        oidx defaulting to zero if not provided.::

            gg(0,0,1000)   # triplet addressing to remainder volumes, NB all are ridx:0 pidx:0 
            gg(2792)       # same volume via node indexing 

            gg(5,0,2)      # first placement of cathode volume (DYB)
            gg(3201)       # same volume via node indexing   

            gg(5,671,2)    # last placement of cathode volume (DYB)
            gg(11410)      # same volume via node indexing 

        """
        log.debug("args %s len(args) %d " % (str(args), len(args))) 

        nidxs = []

        if len(args) == 1: 
            nidx = args[0]
            nidxs.append(nidx)
        elif len(args) == 2 or len(args) == 3:
            if len(args) == 2:
                a_ridx,a_pidx = args
                a_oidx = "*"
            elif len(args) == 3:
                a_ridx,a_pidx,a_oidx = args
            else:
                assert 0 
            pass 
            log.debug("a_ridx %s a_pidx %s a_oidx %s" % (a_ridx,a_pidx,a_oidx)) 

            if type(a_ridx) is int:
                ridxs = [a_ridx]
            elif ":" in a_ridx:
                ridxs = range(*map(int,a_ridx.split(":")))
            else:
                assert 0, a_ridx
            pass
          
            log.debug("ridxs %s " % str(ridxs))

            if type(a_pidx) is int:
                pidxs = [a_pidx]
            elif ":" in a_pidx:
                pidxs = range(*map(int,a_pidx.split(":")))
            else:
                assert 0, a_pidx
            pass
 

            for ridx in ridxs:
                if a_oidx == "*":
                    num_volumes = self.get_num_volumes(ridx)
                    oidxs = range(num_volumes)
                else:
                    oidxs = [a_oidx]
                pass 
                for pidx in pidxs:
                    for oidx in oidxs:
                        nidx = self.get_node_index(ridx,pidx,oidx)
                        nidxs.append(nidx)
                    pass
                pass
            pass
        else:
            assert 0, "expecting argument of 1/2/3 integers"
        pass

        for nidx in nidxs:
            if self.args.nidx:
                print(nidx)
            elif self.args.brief:
                self.brief(nidx) 
            elif self.args.names:
                self.names(nidx) 
            elif self.args.sonames:
                self.sonames(nidx) 
            elif self.args.soidx:
                self.soidx(nidx) 
            else: 
                self.dump_node(nidx) 
            pass
        pass

        if self.suppress_count > 0:
            log.info("supressed %d volumes " % self.suppress_count )
        pass


    def idsmry(self, nidx):
        """
        volume identity summary 
        """
        iden = self.all_volume_identity[nidx]
        nidx2,triplet,shape,_ = iden  
        assert nidx == nidx2
        sidx = iden[-1].view(np.int32)  
        iden_s = "nidx:{nidx:6d} triplet:{triplet:8x} sh:{sh:6x} sidx:{sidx:5d} ".format(nidx=nidx,triplet=triplet,sh=shape,sidx=sidx)

        nrpo = self.nrpo[nidx]
        nrpo_s = "nrpo( %6d %5d %5d %5d )" % tuple(nrpo)

        midx = self.midx[nidx] 
        bidx = self.bidx[nidx] 
        shape_s = "shape( %3d %3d  %40s %40s)" % (midx,bidx, self.mlibnames[nidx], self.blibnames[nidx] )
        print( "%s  %s  %s " % (iden_s, nrpo_s, shape_s) )

    def names(self, nidx):
        pv = self.pv[nidx].decode('utf-8')  

        lv = self.lv[nidx].decode('utf-8')  
        midx = self.midx[nidx] 
        nrpo = self.nrpo[nidx]
        msn = self.msn[midx]

        would_suppress = self.Suppress(pv) or self.Suppress(lv) or self.Suppress(msn)
        suppress = self.args.suppress and would_suppress
        sup = "S" if would_suppress else " "
        if suppress:
            self.suppress_count += 1 
        else: 
            nrpo_s = "nrpo( %6d %5d %5d %5d )" % tuple(nrpo)
            print( "%1s %s %50s %50s  %3d %s " % (sup, nrpo_s, pv, lv, midx, msn) )
        pass  

    def sonames(self, nidx):
        midx = self.midx[nidx] 
        msn = self.msn[midx]
        if args.terse:
            print(msn)
        elif args.errout:
            print(midx,file=sys.stdout)
            print(msn, file=sys.stderr)
        else:
            print( "%3d %s " % (midx, msn) )
        pass

    def soidx(self, nidx):
        midx = self.midx[nidx] 
        msn = self.msn[midx]
        if args.terse:
            print(midx)
        elif args.errout:
            print(midx,file=sys.stdout)
            print(msn, file=sys.stderr)
        else:
            print( "%3d %s " % (midx, msn) )
        pass


    def bbsmry(self, nidx):
        gg = self
        bb = gg.all_volume_bbox[nidx]
        ce = gg.all_volume_center_extent[nidx]
        print(" {nidx:5d} {ce!s:20s} ".format(nidx=nidx,bb=bb,ce=ce))

    def brief(self,nidx):
        gg = self
        gg.idsmry(nidx)

    def dump_node(self,nidx):
        gg = self
        gg.idsmry(nidx)

        #print("gg.mlibnames[%d] : %s " % (i, gg.mlibnames[nidx]) )
        #print("gg.blibnames[%d] : %s " % (i, gg.blibnames[nidx]) )

        bb = gg.all_volume_bbox[nidx]
        ce = gg.all_volume_center_extent[nidx]
        c4 = ce.copy()
        c4[3] = 1.
        
        tr = gg.all_volume_transforms[nidx]
        it = np.linalg.inv(tr) 
        ibb = np.dot( bb, it )   ## apply inverse transform to the volumes bbox (mn,mx), should give symmetric (mn,mx)   
        cbb = (bb[0]+bb[1])/2.   ## center of bb should be same as c4
        assert np.allclose( c4, cbb )

        ic4 = np.dot( c4, it )   ## should be close to origin
        gt = gg.all_volume_transforms[nidx]  

        print("\ngt : gg.all_volume_transforms[%d]" % nidx)
        print(gt)
        print("\ntr : transform")
        print(tr)
        print("\nit : inverted transform")
        print(it)
        print("\nbb : bbox4")
        print(bb)
        print("\ncbb : (bb[0]+bb[1])/2.")
        print(cbb)
        print("\nc4 : center4")
        print(c4)
        print("\nce : center_extent")
        print(ce)
        print("\nic4 : np.dot( c4, it) : inverse transform applied to center4 : expect close to origin ")
        print(ic4)
        print("\nibb : np.dot( bb, it) : inverse transform applied to bbox4 : expect symmetric around origin")
        print(ibb)

    def consistency_check(self):
        gg = self
        log.info("consistency_check")
        gg.summary()

        tr = gg.get_all_transforms()



def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(     "idx", nargs="*", help="Node index or triplet index of form \"1/0/0\" or \"1/\" to dump.")
    parser.add_argument(     "--nidx", default=False, action="store_true", help="Dump only the node index, useful when the input is triplet index." ) 
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    parser.add_argument(  "-c","--check", action="store_true", help="Consistency check" ) 
    parser.add_argument(  "-i","--idsmry", action="store_true", help="Slice identity summary interpreting idx as slice range." ) 
    parser.add_argument(  "-b","--bbsmry", action="store_true", help="Slice bbox summary interpreting idx as slice range." ) 
    parser.add_argument(  "--brief", action="store_true", help="Brief summary of nodes selected by idx." ) 
    parser.add_argument(  "--names", action="store_true", help="Identity and PV/LV/SO names of  nodes selected by idx." ) 
    parser.add_argument(  "--mm",   action="store_true", help="MM Names" ) 
    parser.add_argument(  "--mmsmry",   action="store_true", help="MM Names" ) 
    parser.add_argument(  "--sonames", action="store_true", help="Dump solid names for the nodes selected by idx." ) 
    parser.add_argument(  "--soidx", action="store_true", help="Dump solid_idx (aka: lvidx or meshidx/midx) for the nodes selected by node or triplet idx." ) 
    parser.add_argument(  "--suppress", action="store_true", help="Suppress dumping/listing volumes with names including a list of suppressed strings provided by envvar OPTICKS_GGEO_SUPPRESS")
    parser.add_argument(  "-t","--terse", action="store_true", help="Terse output" ) 
    parser.add_argument(  "-s","--errout", action="store_true", help="Split output writing to both stderr and stdout" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    if len(args.idx) == 0:
        args.idx = [0]
    pass
    return args  


def misc(gg):
    bbox = gg.all_volume_bbox
    print("gg.all_volume_bbox\n",bbox)
    t000 = gg.get_transform(0,0,0)
    print("t000\n",t000)

def triplet_(rpo):
    elem = []
    for s in rpo.split("/"):
        if s == "*" or ":" in s:
            elem.append(s)
        else:
            if s.isnumeric():
                elem.append(int(s))
            else:
                elem.append(0)
            pass
        pass
    pass
    return elem 


if __name__ == '__main__':
    args = parse_args(__doc__)
    gg = GGeo(args)

    if args.suppress:
        log.info("using suppression %s specificed by envvar OPTICKS_GGEO_SUPPRESS " % gg.SUPPRESS_PTN ) 
    else:
        log.info("suppression %s specificed by envvar OPTICKS_GGEO_SUPPRESS can be enabled with --suppression option " % gg.SUPPRESS_PTN ) 
    pass

    if args.check:
        gg.consistency_check()
    elif args.mm:
        for ridx in gg.mmidx:
            pidx = 0 
            oidx = 0 
            num_vol = gg.get_num_volumes(ridx) 
            msn = gg.get_soname(ridx, pidx, oidx)
            mm_name = "%d:%s" % (num_vol, msn)
            print(mm_name)
        pass
    elif args.mmsmry:


        labels = ["ridx","plc","vol", "component name", "note"]
        hfmt = ["%4s", "%6s", "%5s", "%-40s", "%-25s" ]
        rfmt = ["%4d", "%6d", "%5d", "%-40s", "%-25s" ]
        wids = [4,  6, 5, 40 , 25 ]
        pre  =  ["",  "",   "", "   ", "  " ]

        mm_notes = {}
        mm_notes[0] = "non-repeated remainder"
        mm_notes[9] = "repeated parts of TT"

        t = np.empty( [len(gg.mmidx), len(labels)], dtype=np.object ) 

        for i, ridx in enumerate(gg.mmidx):
            pidx = 0 
            oidx = 0 
            num_vol = gg.get_num_volumes(ridx) 
            num_plc = gg.get_num_placements(ridx) 
            msn = gg.get_soname(ridx, pidx, oidx)
            mm_name = "%d:%s" % (num_vol, msn)
            mm_note = mm_notes.get(i, "")
            row = tuple([ridx, num_plc, num_vol, mm_name, mm_note])
            line = " ".join(rfmt) % row
            print(line)
            t[i] = row 
        pass
        rst = RSTTable.Render(t, labels, wids, hfmt, rfmt, pre )
        print(rst) 
        pass
    elif args.idsmry or args.bbsmry:

        beg = int(args.idx[0])
        end = int(args.idx[1]) if len(args.idx) > 1 else min(gg.num_volumes,int(args.idx[0])+50) 
        for idx in list(range(beg, end)):
            if args.idsmry:
                gg.idsmry(idx)
            elif args.bbsmry:
                gg.bbsmry(idx)
            else:
                pass
            pass
        pass
    else:
        for idx in args.idx:
            if str(idx).isnumeric():    
                gg(int(idx))
            else:
                gg(*triplet_(idx))
            pass  
        pass
    pass

 
