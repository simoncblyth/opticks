OpSnapTest_debug_slowdown_with_new_geometry
=============================================


Need to remake geocache with the pinned down ordering 
---------------------------------------------------------

Before::

    O[blyth@localhost opticks]$ ./ana/GParts.sh 
    /home/blyth/junotop/ExternalLibs/Opticks/0.0.0-rc1/bashrc: line 4: /home/blyth/junotop/ExternalLibs/Opticks/0.0.0-rc1/bin/opticks-setup.sh: No such file or directory
    mo .bashrc OPTICKS_MODE:dev O : ordinary opticks dev ontop of juno externals CMTEXTRATAGS:
    python3 ./ana/GParts.py
    Solid 0 : /tmp/blyth/opticks/GParts/0 : primbuf (3084, 4) partbuf (17346, 4, 4) tranbuf (7917, 3, 4, 4) idxbuf (3084, 4) 
    Solid 1 : /tmp/blyth/opticks/GParts/1 : primbuf (5, 4) partbuf (7, 4, 4) tranbuf (5, 3, 4, 4) idxbuf (5, 4) 
    Solid 2 : /tmp/blyth/opticks/GParts/2 : primbuf (6, 4) partbuf (30, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 
    Solid 3 : /tmp/blyth/opticks/GParts/3 : primbuf (6, 4) partbuf (54, 4, 4) tranbuf (29, 3, 4, 4) idxbuf (6, 4) 
    Solid 4 : /tmp/blyth/opticks/GParts/4 : primbuf (6, 4) partbuf (28, 4, 4) tranbuf (15, 3, 4, 4) idxbuf (6, 4) 
    Solid 5 : /tmp/blyth/opticks/GParts/5 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (11, 3, 4, 4) idxbuf (1, 4) 
    Solid 6 : /tmp/blyth/opticks/GParts/6 : primbuf (1, 4) partbuf (1, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
    Solid 7 : /tmp/blyth/opticks/GParts/7 : primbuf (1, 4) partbuf (31, 4, 4) tranbuf (9, 3, 4, 4) idxbuf (1, 4) 
    Solid 8 : /tmp/blyth/opticks/GParts/8 : primbuf (1, 4) partbuf (3, 4, 4) tranbuf (1, 3, 4, 4) idxbuf (1, 4) 
    Solid 9 : /tmp/blyth/opticks/GParts/9 : primbuf (130, 4) partbuf (130, 4, 4) tranbuf (130, 3, 4, 4) idxbuf (130, 4) 





* mv /tmp/blyth/opticks/okop/OpFlightPathTest/FlightPath.mp4 ~/Movies/lFasteners_phys_~5,_256.mp4  : 0.085    # factor 20 from skipping the conical cover 
* mv /tmp/blyth/opticks/okop/OpFlightPathTest/FlightPath.mp4 ~/Movies/lFasteners_phys_~0_256.mp4   : 1.720    

::

    O[blyth@localhost opticks]$ PERIOD=8 EMM=~5,6 PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1 

    mv /tmp/blyth/opticks/okop/OpFlightPathTest/FlightPath.mp4 ~/Movies/lFasteners_phys_~5,6_256.mp4

    O[blyth@localhost opticks]$ PERIOD=8 EMM=~5,6,7,8 PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1  

    O[blyth@localhost opticks]$ PERIOD=8 EMM=~5,6,7,8,9 PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1 

    O[blyth@localhost opticks]$ PERIOD=8 EMM=0, PVN=lFasteners_phys flight.sh --rtx 1 --cvd 1 




The 590s are the problem 
---------------------------

::

       590 :                                 lAddition0x35ff5f0 : lAddition_phys0x35ff770 lAddition_phys0x35ff870 lAddition_phys0x35ff970 
       590 :                                lFasteners0x34cdf00 : lFasteners_phys0x34ce040 lFasteners_phys0x34ce140 lFasteners_phys0x35750f0 
       590 :                                    lSteel0x352c760 : lSteel_phys0x352c890 lSteel_phys0x352a4a0 lSteel_phys0x352a560 
       590 :                                    lUpper0x35b5a00 : lUpper_phys0x35b5ac0 lUpper_phys0x35b5bb0 lUpper_phys0x35b5ca0 


::

    epsilon:~ blyth$  GNodeLib.py --pv lAddition_phys0x --ce
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.pv:lAddition_phys0x matched 590 nodes 
    slice 0:10:1 
    [70258 70259 70260 70261 70262 70263 70264 70265 70266 70267]
    [ 1021.952  1406.597 17789.584   447.067]
    [-1021.952  1406.597 17789.584   447.067]
    [-1653.554  -537.272 17789.584   449.297]
    [    0.    -1738.649 17789.584   450.   ]
    [ 1653.554  -537.272 17789.584   449.297]
    [ 3563.009   374.487 17496.676   447.595]
    [ 3102.653  1791.318 17496.676   447.989]
    [ 2105.82   2898.413 17496.676   445.171]
    [  744.872  3504.346 17496.676   449.008]
    [ -744.872  3504.346 17496.676   449.008]
    epsilon:~ blyth$ 

    epsilon:~ blyth$  GNodeLib.py --pv lFasteners_phys0x --ce
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.pv:lFasteners_phys0x matched 590 nodes 
    slice 0:10:1 
    [69078 69079 69080 69081 69082 69083 69084 69085 69086 69087]
    [ 1023.263  1408.401 17807.176   205.183]
    [-1023.263  1408.401 17807.176   205.183]
    [-1655.674  -537.961 17807.176   206.05 ]
    [    0.    -1740.879 17807.176   206.2  ]
    [ 1655.674  -537.961 17807.176   206.05 ]
    [ 3567.579   374.968 17525.584   205.218]
    [ 3106.632  1793.615 17525.584   205.851]
    [ 2108.521  2902.13  17525.584   204.66 ]
    [  745.827  3508.841 17525.584   205.983]
    [ -745.827  3508.841 17525.584   205.983]
    epsilon:~ blyth$ 



    # why 960 ? 

    epsilon:~ blyth$  GNodeLib.py --pv lSteel_phys0x --ce
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.pv:lSteel_phys0x matched 960 nodes 
    slice 0:10:1 
    [67898 67899 67900 67901 67902 67903 67904 67905 67906 67907]
    [ 1088.691  1498.455 18890.113   903.633]
    [-1088.691  1498.455 18890.113   903.633]
    [-1761.539  -572.359 18890.113   903.633]
    [    0.    -1852.192 18890.113   903.633]
    [ 1761.539  -572.359 18890.113   903.633]
    [ 3795.692   398.943 18593.023   893.887]
    [ 3305.272  1908.3   18593.023   893.887]
    [ 2243.341  3087.694 18593.023   893.887]
    [  793.516  3733.198 18593.023   893.887]
    [ -793.516  3733.198 18593.023   893.887]
    epsilon:~ blyth$ 

    epsilon:GNodeLib blyth$ grep lSteel_phys0x all_volume_PVNames.txt | wc -l
         960

    In [3]: nlib.pvfind("lSteel_phys0x").shape
    Out[3]: (960,)

    In [14]: np.diff(pvi[-590:])    ## two contiguous blocks 
    Out[14]:
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1




    epsilon:~ blyth$ GNodeLib.py --pv lUpper_phys0x --ce
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.pv:lUpper_phys0x matched 590 nodes 
    slice 0:10:1 
    [69668 69669 69670 69671 69672 69673 69674 69675 69676 69677]
    [ 1030.576  1418.466 17918.443   194.088]
    [-1030.576  1418.466 17918.443   194.088]
    [-1667.507  -541.806 17918.443   194.884]
    [    0.    -1753.321 17918.443   195.   ]
    [ 1667.507  -541.806 17918.443   194.884]
    [ 3593.076   377.648 17630.072   194.089]
    [ 3128.835  1806.434 17630.072   194.758]
    [ 2123.59   2922.871 17630.072   193.648]
    [  751.157  3533.918 17630.072   194.832]
    [ -751.157  3533.918 17630.072   194.832]
    epsilon:~ blyth$ 












::


     PVN=lFasteners_phys EMM=0,1,2,3,4,5,6,7,8,9 flightpath.sh --rtx 1 --cvd 1 --flightpathscale=3
     mv /tmp/blyth/opticks/okop/OpFlightPathTest/FlightPath.mp4 ~/Movies/all_lFasteners_phys_FlightPath.mp4


     PVN=lFasteners_phys EMM=5,6,7,8 flightpath.sh --rtx 1 --cvd 1 --flightpathscale=3
     mv /tmp/blyth/opticks/okop/OpFlightPathTest/FlightPath.mp4 ~/Movies/all_590_emm_5,6,7,8_FlightPath.mp4


::

    O[blyth@localhost OpFlightPathTest]$ PVN=lFasteners_phys EMM=5 flightpath.sh --rtx 1 --cvd 1

    epsilon:tests blyth$ mv /tmp/blyth/opticks/okop/OpFlightPathTest/FlightPath.mp4 ~/Movies/temple_inside_FlightPath.mp4


    O[blyth@localhost OpFlightPathTest]$ PVN=lFasteners_phys EMM=5 EYE=-3,-3,-3 flightpath.sh --rtx 1 --cvd 1

    ## ahha, changing eye does nothing for flightpath

    O[blyth@localhost OpFlightPathTest]$ PVN=lFasteners_phys EMM=5 flightpath.sh --flightpathscale=3 --rtx 1 --cvd 1

    mv /tmp/blyth/opticks/okop/OpFlightPathTest/FlightPath.mp4 ~/Movies/flying_saucer_outside_FlightPath.mp4





::

    In [4]: w = np.where(nlib.lvidx == 96 )

    In [5]: nlib.pv[w]
    Out[5]:
    array([b'lAddition_phys0x35ff770', b'lAddition_phys0x35ff870', b'lAddition_phys0x35ff970', b'lAddition_phys0x35ffa70', b'lAddition_phys0x3655ba0', b'lAddition_phys0x3655ca0',
           b'lAddition_phys0x3655da0', b'lAddition_phys0x3655ea0', b'lAddition_phys0x3655fa0', b'lAddition_phys0x36560a0', b'lAddition_phys0x36561a0', b'lAddition_phys0x36562a0',
           b'lAddition_phys0x36563a0', b'lAddition_phys0x36564a0', b'lAddition_phys0x36565a0', b'lAddition_phys0x36566a0', b'lAddition_phys0x36567a0', b'lAddition_phys0x36568a0',
           b'lAddition_phys0x36569a0', b'lAddition_phys0x3656aa0', b'lAddition_phys0x3656ba0', b'lAddition_phys0x3656ca0', b'lAddition_phys0x3656da0', b'lAddition_phys0x3656ea0',
           b'lAddition_phys0x3656fa0', b'lAddition_phys0x36570a0', b'lAddition_phys0x36571a0', b'lAddition_phys0x36572a0', b'lAddition_phys0x36573a0', b'lAddition_phys0x36574a0',


    In [10]: nlib.pv[np.where(nlib.lvidx == 93 )].shape                                                                                                                                      
    Out[10]: (590,)

    In [11]: nlib.pv[np.where(nlib.lvidx == 94 )].shape                                                                                                                                      
    Out[11]: (590,)

    In [12]: nlib.pv[np.where(nlib.lvidx == 95 )].shape                                                                                                                                      
    Out[12]: (590,)

    In [13]: nlib.pv[np.where(nlib.lvidx == 96 )].shape                                                                                                                                      
    Out[13]: (590,)



    In [15]: nlib.pv[np.where(nlib.lvidx == 93 )][:3]                                                                                                                                        
    Out[15]: array([b'lSteel_phys0x352c890', b'lSteel_phys0x352a4a0', b'lSteel_phys0x352a560'], dtype='|S100')

    In [16]: nlib.pv[np.where(nlib.lvidx == 94 )][:3]                                                                                                                                        
    Out[16]: array([b'lFasteners_phys0x34ce040', b'lFasteners_phys0x34ce140', b'lFasteners_phys0x35750f0'], dtype='|S100')

    In [17]: nlib.pv[np.where(nlib.lvidx == 95 )][:3]                                                                                                                                        
    Out[17]: array([b'lUpper_phys0x35b5ac0', b'lUpper_phys0x35b5bb0', b'lUpper_phys0x35b5ca0'], dtype='|S100')

    In [18]: nlib.pv[np.where(nlib.lvidx == 96 )][:3]                                                                                                                                        
    Out[18]: array([b'lAddition_phys0x35ff770', b'lAddition_phys0x35ff870', b'lAddition_phys0x35ff970'], dtype='|S100')


    epsilon:GItemList blyth$ cat.py -s 89,90,91,92,93,94,95,96,97,98 GMeshLib.txt 
    89   90   sTarget0x34fe8a0
    90   91   sAcrylic0x34fe230
    91   92   sStrut0x3501680
    92   93   sStrut0x3559670

    93   94   sStrutBallhead0x352a360
    94   95   uni10x34cdcb0
    95   96   base_steel0x360d8f0
    96   97   uni_acrylic30x35ff3d0

    97   98   solidXJanchor0x363f2f0
    98   99   NNVTMCPPMTsMask0x3c9fa80
    epsilon:GItemList blyth$ 









::

    PVN=lFasteners_phys EMM=5 snap.sh         ## dont include the address in PVN, it keeps changing


    2021-04-19 05:43:06.682 INFO  [340264] [OGeo::convert@302] [ nmm 10
    2021-04-19 05:43:06.682 ERROR [340264] [OGeo::convert@313] MergedMesh 0 IS NOT ENABLED 
    2021-04-19 05:43:06.682 ERROR [340264] [OGeo::convert@313] MergedMesh 1 IS NOT ENABLED 
    2021-04-19 05:43:06.682 ERROR [340264] [OGeo::convert@313] MergedMesh 2 IS NOT ENABLED 
    2021-04-19 05:43:06.682 ERROR [340264] [OGeo::convert@313] MergedMesh 3 IS NOT ENABLED 
    2021-04-19 05:43:06.682 ERROR [340264] [OGeo::convert@313] MergedMesh 4 IS NOT ENABLED 
    2021-04-19 05:43:06.756 ERROR [340264] [OGeo::convert@313] MergedMesh 6 IS NOT ENABLED 
    2021-04-19 05:43:06.756 ERROR [340264] [OGeo::convert@313] MergedMesh 7 IS NOT ENABLED 
    2021-04-19 05:43:06.756 ERROR [340264] [OGeo::convert@313] MergedMesh 8 IS NOT ENABLED 
    2021-04-19 05:43:06.756 ERROR [340264] [OGeo::convert@313] MergedMesh 9 IS NOT ENABLED 
    2021-04-19 05:43:06.756 INFO  [340264] [OGeo::convert@321] ] nmm 10
    2021-04-19 05:43:06.758 INFO  [340264] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
    2021-04-19 05:43:06.758 INFO  [340264] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix snap-emm-5-
    2021-04-19 05:43:06.758 ERROR [340264] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn 69078 cmdline_target 0 gdmlaux_target -1 active_target 69078
    2021-04-19 05:43:06.758 INFO  [340264] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-72.5279) front 0.5774,0.5774,0.5774
     count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/snap-emm-5-00000.jpg dt     1.8817
    2021-04-19 05:43:10.114 INFO  [340264] [OTracer::report@192] OpTracer::snap
    2021-04-19 05:43:10.114 INFO  [340264] [OTracer::report@195] 
     trace_count              1 trace_prep         0.00054 avg    0.00054
     trace_time         3.26851 avg    3.26851

    2021-04-19 05:43:10.114 INFO  [340264] [OTracer::report@203] OTracer::report
                  validate000                 0.000387
                   compile000              4.99999e-06
                 prelaunch000                  1.38532
                    launch000                  1.88166
                    launchAVG                  1.88166

    2021-04-19 05:43:10.114 INFO  [340264] [OTracer::report@208] save to /home/blyth/local/opticks/results/OpSnapTest/R0_cvd_/20210419_054304
    2021-04-19 05:43:10.114 INFO  [340264] [BFile::preparePath@844] created directory /home/blyth/local/opticks/results/OpSnapTest/R0_cvd_/20210419_054304
    2021-04-19 05:43:10.115 INFO  [340264] [OpTracer::snap@182] ]
    rc 0





::

    epsilon:ana blyth$ ipython -i -- GNodeLib.py --ulv --detail
    Key.v9:OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
    /usr/local/opticks/geocache/OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1
    args.ulv found 131 unique LV names
    GLb1.bt02_HBeam0x34c1e00
    GLb1.bt05_HBeam0x34cf620
    GLb1.bt06_HBeam0x34d1e20
    GLb1.bt07_HBeam0x34d4620
    GLb1.bt08_HBeam0x34d6e20
    GLb1.up01_HBeam0x34ba600
    GLb1.up02_HBeam0x34b7e00
    GLb1.up03_HBeam0x34b5600
    GLb1.up04_HBeam0x34b2e00
    GLb1.up05_HBeam0x3487c90
    unique lv in descending count order, with names of corresponding pv 
         32256 :                                      lBar0x4ee75d0 : pBar0x4ef4970 pBar0x4ef4970 pBar0x4ef4970 
         32256 :                                  lCoating0x4ee7440 : pCoating_00_0x4ef1ef0 pCoating_01_0x4ef1f90 pCoating_02_0x4ef2030 
         25600 :                        PMT_3inch_body_log0x4436ce0 : PMT_3inch_body_phys0x4437230 PMT_3inch_body_phys0x4437230 PMT_3inch_body_phys0x4437230 
         25600 :                        PMT_3inch_cntr_log0x4437120 : PMT_3inch_cntr_phys0x4437410 PMT_3inch_cntr_phys0x4437410 PMT_3inch_cntr_phys0x4437410 
         25600 :                      PMT_3inch_inner1_log0x4436f00 : PMT_3inch_inner1_phys0x44372b0 PMT_3inch_inner1_phys0x44372b0 PMT_3inch_inner1_phys0x44372b0 
         25600 :                      PMT_3inch_inner2_log0x4437010 : PMT_3inch_inner2_phys0x4437360 PMT_3inch_inner2_phys0x4437360 PMT_3inch_inner2_phys0x4437360 
         25600 :                             PMT_3inch_log0x4436df0 : PMT_3inch_log_phys0x4437d00 PMT_3inch_log_phys0x4437e00 PMT_3inch_log_phys0x4437f00 
         12612 :            NNVTMCPPMT_PMT_20inch_body_log0x3caeb60 : NNVTMCPPMT_PMT_20inch_body_phys0x3caefa0 NNVTMCPPMT_PMT_20inch_body_phys0x3caefa0 NNVTMCPPMT_PMT_20inch_body_phys0x3caefa0 
         12612 :          NNVTMCPPMT_PMT_20inch_inner1_log0x3caed60 : NNVTMCPPMT_PMT_20inch_inner1_phys0x3caf030 NNVTMCPPMT_PMT_20inch_inner1_phys0x3caf030 NNVTMCPPMT_PMT_20inch_inner1_phys0x3caf030 
         12612 :          NNVTMCPPMT_PMT_20inch_inner2_log0x3caee80 : NNVTMCPPMT_PMT_20inch_inner2_phys0x3caf0f0 NNVTMCPPMT_PMT_20inch_inner2_phys0x3caf0f0 NNVTMCPPMT_PMT_20inch_inner2_phys0x3caf0f0 
         12612 :                 NNVTMCPPMT_PMT_20inch_log0x3caec40 : NNVTMCPPMT_PMT_20inch_log_phys0x3c9fe80 NNVTMCPPMT_PMT_20inch_log_phys0x3c9fe80 NNVTMCPPMT_PMT_20inch_log_phys0x3c9fe80 
         12612 :                           NNVTMCPPMTlMask0x3c9fc80 : NNVTMCPPMTpMask0x3c9fe00 NNVTMCPPMTpMask0x3c9fe00 NNVTMCPPMTpMask0x3c9fe00 
         12612 :                    NNVTMCPPMTlMaskVirtual0x3cb41a0 : pLPMT_NNVT_MCPPMT0x3cbba60 pLPMT_NNVT_MCPPMT0x3cbbbb0 pLPMT_NNVT_MCPPMT0x3cb97c0 
          5000 :       HamamatsuR12860_PMT_20inch_body_log0x3c93830 : HamamatsuR12860_PMT_20inch_body_phys0x345b3c0 HamamatsuR12860_PMT_20inch_body_phys0x345b3c0 HamamatsuR12860_PMT_20inch_body_phys0x345b3c0 
          5000 :     HamamatsuR12860_PMT_20inch_inner1_log0x345b160 : HamamatsuR12860_PMT_20inch_inner1_phys0x3c94040 HamamatsuR12860_PMT_20inch_inner1_phys0x3c94040 HamamatsuR12860_PMT_20inch_inner1_phys0x3c94040 
          5000 :     HamamatsuR12860_PMT_20inch_inner2_log0x345b290 : HamamatsuR12860_PMT_20inch_inner2_phys0x3c94100 HamamatsuR12860_PMT_20inch_inner2_phys0x3c94100 HamamatsuR12860_PMT_20inch_inner2_phys0x3c94100 
          5000 :            HamamatsuR12860_PMT_20inch_log0x3c93920 : HamamatsuR12860_PMT_20inch_log_phys0x3c9b3b0 HamamatsuR12860_PMT_20inch_log_phys0x3c9b3b0 HamamatsuR12860_PMT_20inch_log_phys0x3c9b3b0 
          5000 :                      HamamatsuR12860lMask0x3c9b1a0 : HamamatsuR12860pMask0x3c9b320 HamamatsuR12860pMask0x3c9b320 HamamatsuR12860pMask0x3c9b320 
          5000 :               HamamatsuR12860lMaskVirtual0x3c9a5c0 : pLPMT_Hamamatsu_R128600x3cbbae0 pLPMT_Hamamatsu_R128600x3cb98c0 pLPMT_Hamamatsu_R128600x3cb9cc0 
          2400 :                  PMT_20inch_veto_body_log0x3ca5360 : PMT_20inch_veto_body_phys0x3ca57a0 PMT_20inch_veto_body_phys0x3ca57a0 PMT_20inch_veto_body_phys0x3ca57a0 
          2400 :                PMT_20inch_veto_inner1_log0x3ca5580 : PMT_20inch_veto_inner1_phys0x3ca5820 PMT_20inch_veto_inner1_phys0x3ca5820 PMT_20inch_veto_inner1_phys0x3ca5820 
          2400 :                PMT_20inch_veto_inner2_log0x3ca5690 : PMT_20inch_veto_inner2_phys0x3ca58d0 PMT_20inch_veto_inner2_phys0x3ca58d0 PMT_20inch_veto_inner2_phys0x3ca58d0 
          2400 :                       PMT_20inch_veto_log0x3ca5470 : PMT_20inch_veto_log_phys0x3ca5fa0 PMT_20inch_veto_log_phys0x3ca5fa0 PMT_20inch_veto_log_phys0x3ca5fa0 
          2400 :                 mask_PMT_20inch_vetolMask0x3ca1cb0 : mask_PMT_20inch_vetopMask0x3ca1e40 mask_PMT_20inch_vetopMask0x3ca1e40 mask_PMT_20inch_vetopMask0x3ca1e40 
          2400 :          mask_PMT_20inch_vetolMaskVirtual0x3ca10e0 : mask_PMT_20inch_vetolMaskVirtual_phys0x4433460 mask_PMT_20inch_vetolMaskVirtual_phys0x4dd9ec0 mask_PMT_20inch_vetolMaskVirtual_phys0x4dd9fd0 

           590 :                                 lAddition0x35ff5f0 : lAddition_phys0x35ff770 lAddition_phys0x35ff870 lAddition_phys0x35ff970 
           590 :                                lFasteners0x34cdf00 : lFasteners_phys0x34ce040 lFasteners_phys0x34ce140 lFasteners_phys0x35750f0 
           590 :                                    lSteel0x352c760 : lSteel_phys0x352c890 lSteel_phys0x352a4a0 lSteel_phys0x352a560 
           590 :                                    lUpper0x35b5a00 : lUpper_phys0x35b5ac0 lUpper_phys0x35b5bb0 lUpper_phys0x35b5ca0 

           504 :                                    lPanel0x4ee7120 : pPanel_0_f_0x4ef1b70 pPanel_1_f_0x4ef1c10 pPanel_2_f_0x4ef1cb0 
           504 :                                lPanelTape0x4ee72b0 : pPanelTape0x4ef1e50 pPanelTape0x4ef1e50 pPanelTape0x4ef1e50 
           370 :                                    lSteel0x3501790 : lSteel_phys0x34fd1c0 lSteel_phys0x3501920 lSteel_phys0x3501a40 
           220 :                                   lSteel20x3559780 : lSteel2_phys0x3559810 lSteel2_phys0x3557440 lSteel2_phys0x3557530 
           126 :                                  lPlanef_0x4ee7010 : pPlane_0_ff_0x4ee76d0 pPlane_1_ff_0x4ef1ad0 pPlane_0_ff_0x4ee76d0 
            64 :                                lXJfixture0x3645b00 : lXJfixture_phys0x3652450 lXJfixture_phys0x36524d0 lXJfixture_phys0x36525a0 
            63 :                                  lWallff_0x4ee6df0 : pWall_000_0x4ee77e0 pWall_001_0x4ee6f90 pWall_002_0x4ee7bb0 
            56 :                                 lXJanchor0x363f540 : lXJanchor_phys0x363f6c0 lXJanchor_phys0x363f7c0 lXJanchor_phys0x363f8c0 
            36 :                                lSJFixture0x364dd80 : lSJFixture_phys0x364df00 lSJFixture_phys0x364e030 lSJFixture_phys0x3649a10 
            30 :                           GLb1.bt02_HBeam0x34c1e00 : GLb1.bt02_HBeam_phys0x34c1f90 GLb1.bt02_HBeam_phys0x34c2070 GLb1.bt02_HBeam_phys0x34c2180 
            30 :                           GLb1.bt05_HBeam0x34cf620 : GLb1.bt05_HBeam_phys0x34cf7b0 GLb1.bt05_HBeam_phys0x34cf890 GLb1.bt05_HBeam_phys0x34cf9a0 
            30 :                           GLb1.bt06_HBeam0x34d1e20 : GLb1.bt06_HBeam_phys0x34d1fb0 GLb1.bt06_HBeam_phys0x34d2090 GLb1.bt06_HBeam_phys0x34d21a0 
            30 :                           GLb1.bt07_HBeam0x34d4620 : GLb1.bt07_HBeam_phys0x34d47b0 GLb1.bt07_HBeam_phys0x34d4890 GLb1.bt07_HBeam_phys0x34d49a0 
            30 :                           GLb1.bt08_HBeam0x34d6e20 : GLb1.bt08_HBeam_phys0x34d6fb0 GLb1.bt08_HBeam_phys0x34d7090 GLb1.bt08_HBeam_phys0x34d71a0 
            30 :                           GLb1.up01_HBeam0x34ba600 : GLb1.up01_HBeam_phys0x34ba790 GLb1.up01_HBeam_phys0x34ba870 GLb1.up01_HBeam_phys0x34ba980 
            30 :                           GLb1.up02_HBeam0x34b7e00 : GLb1.up02_HBeam_phys0x34b7f90 GLb1.up02_HBeam_phys0x34b8070 GLb1.up02_HBeam_phys0x34b8180 
            30 :                           GLb1.up03_HBeam0x34b5600 : GLb1.up03_HBeam_phys0x34b5790 GLb1.up03_HBeam_phys0x34b5870 GLb1.up03_HBeam_phys0x34b5980 
            30 :                           GLb1.up04_HBeam0x34b2e00 : GLb1.up04_HBeam_phys0x34b2f90 GLb1.up04_HBeam_phys0x34b3070 GLb1.up04_HBeam_phys0x34b3180 
            30 :                           GLb1.up05_HBeam0x3487c90 : GLb1.up05_HBeam_phys0x3487e20 GLb1.up05_HBeam_phys0x3487f00 GLb1.up05_HBeam_phys0x3488010 
            30 :                           GLb2.bt01_HBeam0x34bf600 : GLb2.bt01_HBeam_phys0x34bf790 GLb2.bt01_HBeam_phys0x34bf870 GLb2.bt01_HBeam_phys0x34bf980 
            30 :                           GLb2.bt03_HBeam0x345d180 : GLb2.bt03_HBeam_phys0x345d310 GLb2.bt03_HBeam_phys0x345d3f0 GLb2.bt03_HBeam_phys0x345d500 
            30 :                           GLb2.bt04_HBeam0x34972e0 : GLb2.bt04_HBeam_phys0x3497470 GLb2.bt04_HBeam_phys0x3497550 GLb2.bt04_HBeam_phys0x3497660 
            30 :                            GLb2.equ_HBeam0x34bce00 : GLb2.equ_HBeam_phys0x34bcf90 GLb2.equ_HBeam_phys0x34bd070 GLb2.equ_HBeam_phys0x34bd180 
            30 :                           GLb2.up06_HBeam0x34850d0 : GLb2.up06_HBeam_phys0x3485260 GLb2.up06_HBeam_phys0x3485340 GLb2.up06_HBeam_phys0x3485450 
            30 :                           GLb2.up07_HBeam0x34a9680 : GLb2.up07_HBeam_phys0x34a9810 GLb2.up07_HBeam_phys0x34a98f0 GLb2.up07_HBeam_phys0x34a9a00 
            30 :                           GLb2.up08_HBeam0x34a6e80 : GLb2.up08_HBeam_phys0x34a7010 GLb2.up08_HBeam_phys0x34a70f0 GLb2.up08_HBeam_phys0x34a7200 
            30 :                           GLb3.bt09_HBeam0x34d9620 : GLb3.bt09_HBeam_phys0x34d97b0 GLb3.bt09_HBeam_phys0x34d9890 GLb3.bt09_HBeam_phys0x34d99a0 
            30 :                           GLb3.bt10_HBeam0x34dbe20 : GLb3.bt10_HBeam_phys0x34dbfb0 GLb3.bt10_HBeam_phys0x34dc090 GLb3.bt10_HBeam_phys0x34dc1a0 
            30 :                           GLb3.bt11_HBeam0x34de620 : GLb3.bt11_HBeam_phys0x34de7b0 GLb3.bt11_HBeam_phys0x34de890 GLb3.bt11_HBeam_phys0x34de9a0 
            30 :                           GLb3.up09_HBeam0x34a4680 : GLb3.up09_HBeam_phys0x34a4810 GLb3.up09_HBeam_phys0x34a48f0 GLb3.up09_HBeam_phys0x34a4a00 
            30 :                           GLb3.up11_HBeam0x349f680 : GLb3.up11_HBeam_phys0x349f810 GLb3.up11_HBeam_phys0x349f8f0 GLb3.up11_HBeam_phys0x349fa00 
            30 :                           GLb4.up10_HBeam0x34a1e80 : GLb4.up10_HBeam_phys0x34a2010 GLb4.up10_HBeam_phys0x34a20f0 GLb4.up10_HBeam_phys0x34a2200 
            30 :                      GLw1.bt05_bt06_HBeam0x348d550 : GLw1.bt05_bt06_HBeam_phys0x348d6d0 GLw1.bt05_bt06_HBeam_phys0x348d7a0 GLw1.bt05_bt06_HBeam_phys0x348d8a0 
            30 :                      GLw1.bt06_bt07_HBeam0x348fb80 : GLw1.bt06_bt07_HBeam_phys0x348fd00 GLw1.bt06_bt07_HBeam_phys0x348fdd0 GLw1.bt06_bt07_HBeam_phys0x348fed0 
            30 :                      GLw1.bt07_bt08_HBeam0x347c830 : GLw1.bt07_bt08_HBeam_phys0x347c9b0 GLw1.bt07_bt08_HBeam_phys0x345db20 GLw1.bt07_bt08_HBeam_phys0x345dbf0 
            30 :                      GLw1.bt08_bt09_HBeam0x3499800 : GLw1.bt08_bt09_HBeam_phys0x3499980 GLw1.bt08_bt09_HBeam_phys0x3499a50 GLw1.bt08_bt09_HBeam_phys0x3499b50 
            30 :                      GLw1.bt09_bt10_HBeam0x349be30 : GLw1.bt09_bt10_HBeam_phys0x349bfb0 GLw1.bt09_bt10_HBeam_phys0x349c080 GLw1.bt09_bt10_HBeam_phys0x349c180 
            30 :                      GLw1.up01_up02_HBeam0x347b200 : GLw1.up01_up02_HBeam_phys0x347b380 GLw1.up01_up02_HBeam_phys0x347b450 GLw1.up01_up02_HBeam_phys0x347b550 
            30 :                      GLw1.up02_up03_HBeam0x3478bd0 : GLw1.up02_up03_HBeam_phys0x3478d50 GLw1.up02_up03_HBeam_phys0x3478e20 GLw1.up02_up03_HBeam_phys0x3478f20 
            30 :                      GLw1.up03_up04_HBeam0x3475f60 : GLw1.up03_up04_HBeam_phys0x34760e0 GLw1.up03_up04_HBeam_phys0x34761b0 GLw1.up03_up04_HBeam_phys0x34762b0 
            30 :                      GLw1.up04_up05_HBeam0x3473930 : GLw1.up04_up05_HBeam_phys0x3473ab0 GLw1.up04_up05_HBeam_phys0x3473b80 GLw1.up04_up05_HBeam_phys0x3473c80 
            30 :                      GLw1.up05_up06_HBeam0x3471300 : GLw1.up05_up06_HBeam_phys0x3471480 GLw1.up05_up06_HBeam_phys0x3471550 GLw1.up05_up06_HBeam_phys0x3471650 
            30 :                      GLw1.up06_up07_HBeam0x346e8d0 : GLw1.up06_up07_HBeam_phys0x346ea50 GLw1.up06_up07_HBeam_phys0x346eb20 GLw1.up06_up07_HBeam_phys0x346ec20 
            30 :                      GLw1.up07_up08_HBeam0x346bf80 : GLw1.up07_up08_HBeam_phys0x346c100 GLw1.up07_up08_HBeam_phys0x346c1d0 GLw1.up07_up08_HBeam_phys0x346c2d0 
            30 :                      GLw1.up08_up09_HBeam0x3469740 : GLw1.up08_up09_HBeam_phys0x34698c0 GLw1.up08_up09_HBeam_phys0x3469990 GLw1.up08_up09_HBeam_phys0x3469a90 
            30 :                      GLw1.up09_up10_HBeam0x3466f70 : GLw1.up09_up10_HBeam_phys0x34670f0 GLw1.up09_up10_HBeam_phys0x34671c0 GLw1.up09_up10_HBeam_phys0x34672c0 
            30 :                      GLw2.bt03_bt04_HBeam0x3477190 : GLw2.bt03_bt04_HBeam_phys0x3488a70 GLw2.bt03_bt04_HBeam_phys0x3488b40 GLw2.bt03_bt04_HBeam_phys0x3488c40 
            30 :                      GLw2.bt04_bt05_HBeam0x348af20 : GLw2.bt04_bt05_HBeam_phys0x348b0a0 GLw2.bt04_bt05_HBeam_phys0x348b170 GLw2.bt04_bt05_HBeam_phys0x348b270 
            30 :                       GLw2.equ_bt01_HBeam0x3480670 : GLw2.equ_bt01_HBeam_phys0x34807f0 GLw2.equ_bt01_HBeam_phys0x34808c0 GLw2.equ_bt01_HBeam_phys0x34809c0 
            30 :                       GLw2.equ_up01_HBeam0x347e040 : GLw2.equ_up01_HBeam_phys0x347e1c0 GLw2.equ_up01_HBeam_phys0x347e290 GLw2.equ_up01_HBeam_phys0x347e390 
            30 :                      GLw3.bt01_bt02_HBeam0x3482ca0 : GLw3.bt01_bt02_HBeam_phys0x3482e20 GLw3.bt01_bt02_HBeam_phys0x3482ef0 GLw3.bt01_bt02_HBeam_phys0x3482ff0 
            30 :                      GLw3.bt02_bt03_HBeam0x3485630 : GLw3.bt02_bt03_HBeam_phys0x34857b0 GLw3.bt02_bt03_HBeam_phys0x3485880 GLw3.bt02_bt03_HBeam_phys0x3485980 
            30 :                          GZ1.A01_02_HBeam0x34e0e20 : GZ1.A01_02_HBeam_phys0x34e0fb0 GZ1.A01_02_HBeam_phys0x34e1090 GZ1.A01_02_HBeam_phys0x34e11a0 
            30 :                          GZ1.A02_03_HBeam0x34e3620 : GZ1.A02_03_HBeam_phys0x34e37b0 GZ1.A02_03_HBeam_phys0x34e3890 GZ1.A02_03_HBeam_phys0x34e39a0 
            30 :                          GZ1.A03_04_HBeam0x34e5e20 : GZ1.A03_04_HBeam_phys0x34e5fb0 GZ1.A03_04_HBeam_phys0x34e6090 GZ1.A03_04_HBeam_phys0x34e61a0 
            30 :                          GZ1.A04_05_HBeam0x34e8620 : GZ1.A04_05_HBeam_phys0x34e87b0 GZ1.A04_05_HBeam_phys0x34e8890 GZ1.A04_05_HBeam_phys0x34e89a0 
            30 :                          GZ1.A05_06_HBeam0x34eae20 : GZ1.A05_06_HBeam_phys0x34eafb0 GZ1.A05_06_HBeam_phys0x34eb090 GZ1.A05_06_HBeam_phys0x34eb1a0 
            30 :                          GZ1.A06_07_HBeam0x34ed620 : GZ1.A06_07_HBeam_phys0x34ed7b0 GZ1.A06_07_HBeam_phys0x34ed890 GZ1.A06_07_HBeam_phys0x34ed9a0 
            30 :                          GZ1.B01_02_HBeam0x34efe20 : GZ1.B01_02_HBeam_phys0x34effb0 GZ1.B01_02_HBeam_phys0x34f0090 GZ1.B01_02_HBeam_phys0x34f01a0 
            30 :                          GZ1.B02_03_HBeam0x34f2620 : GZ1.B02_03_HBeam_phys0x34f27b0 GZ1.B02_03_HBeam_phys0x34f2890 GZ1.B02_03_HBeam_phys0x34f29a0 
            30 :                          GZ1.B03_04_HBeam0x34c43e0 : GZ1.B03_04_HBeam_phys0x34c4570 GZ1.B03_04_HBeam_phys0x34c4650 GZ1.B03_04_HBeam_phys0x34c4760 
            30 :                          GZ1.B04_05_HBeam0x34c6be0 : GZ1.B04_05_HBeam_phys0x34c6d70 GZ1.B04_05_HBeam_phys0x34c6e50 GZ1.B04_05_HBeam_phys0x34c6f60 
            30 :                          GZ1.B05_06_HBeam0x34af010 : GZ1.B05_06_HBeam_phys0x34af1a0 GZ1.B05_06_HBeam_phys0x34af280 GZ1.B05_06_HBeam_phys0x34af390 
            30 :                          GZ1.B06_07_HBeam0x34ac1a0 : GZ1.B06_07_HBeam_phys0x34ac330 GZ1.B06_07_HBeam_phys0x34ac410 GZ1.B06_07_HBeam_phys0x34ac520 
            30 :                         ZC2.A02_B02_HBeam0x3506ce0 : ZC2.A02_B02_HBeam_phys0x3506e60 ZC2.A02_B02_HBeam_phys0x3506f30 ZC2.A02_B02_HBeam_phys0x3507030 
            30 :                         ZC2.A02_B03_HBeam0x3512bd0 : ZC2.A02_B03_HBeam_phys0x3512d50 ZC2.A02_B03_HBeam_phys0x3512e20 ZC2.A02_B03_HBeam_phys0x3512f20 
            30 :                         ZC2.A03_A03_HBeam0x3492600 : ZC2.A03_A03_HBeam_phys0x3492780 ZC2.A03_A03_HBeam_phys0x3492850 ZC2.A03_A03_HBeam_phys0x3492950 
            30 :                         ZC2.A03_B03_HBeam0x3509310 : ZC2.A03_B03_HBeam_phys0x3509490 ZC2.A03_B03_HBeam_phys0x3509560 ZC2.A03_B03_HBeam_phys0x3509660 
            30 :                         ZC2.A03_B04_HBeam0x3515200 : ZC2.A03_B04_HBeam_phys0x3515380 ZC2.A03_B04_HBeam_phys0x3515450 ZC2.A03_B04_HBeam_phys0x3515550 
            30 :                         ZC2.A04_B04_HBeam0x350b940 : ZC2.A04_B04_HBeam_phys0x350bac0 ZC2.A04_B04_HBeam_phys0x350bb90 ZC2.A04_B04_HBeam_phys0x350bc90 
            30 :                         ZC2.A04_B05_HBeam0x3517830 : ZC2.A04_B05_HBeam_phys0x35179b0 ZC2.A04_B05_HBeam_phys0x3517a80 ZC2.A04_B05_HBeam_phys0x3517b80 
            30 :                         ZC2.A05_A05_HBeam0x3494c30 : ZC2.A05_A05_HBeam_phys0x3494db0 ZC2.A05_A05_HBeam_phys0x3494e80 ZC2.A05_A05_HBeam_phys0x3494f80 
            30 :                         ZC2.A05_B05_HBeam0x350df70 : ZC2.A05_B05_HBeam_phys0x350e0f0 ZC2.A05_B05_HBeam_phys0x350e1c0 ZC2.A05_B05_HBeam_phys0x350e2c0 
            30 :                         ZC2.A05_B06_HBeam0x3519e60 : ZC2.A05_B06_HBeam_phys0x3519fe0 ZC2.A05_B06_HBeam_phys0x351a0b0 ZC2.A05_B06_HBeam_phys0x351a1b0 
            30 :                         ZC2.A06_B06_HBeam0x35105a0 : ZC2.A06_B06_HBeam_phys0x3510720 ZC2.A06_B06_HBeam_phys0x35107f0 ZC2.A06_B06_HBeam_phys0x35108f0 
            30 :                         ZC2.A06_B07_HBeam0x351c490 : ZC2.A06_B07_HBeam_phys0x351c610 ZC2.A06_B07_HBeam_phys0x351c6e0 ZC2.A06_B07_HBeam_phys0x351c7e0 
            30 :                         ZC2.B01_B01_HBeam0x351eac0 : ZC2.B01_B01_HBeam_phys0x351ec40 ZC2.B01_B01_HBeam_phys0x351ed10 ZC2.B01_B01_HBeam_phys0x351ee10 
            30 :                         ZC2.B03_B03_HBeam0x35210f0 : ZC2.B03_B03_HBeam_phys0x3521270 ZC2.B03_B03_HBeam_phys0x3521340 ZC2.B03_B03_HBeam_phys0x3521440 
            30 :                         ZC2.B05_B05_HBeam0x3523720 : ZC2.B05_B05_HBeam_phys0x35238a0 ZC2.B05_B05_HBeam_phys0x3523970 ZC2.B05_B05_HBeam_phys0x3523a70 
            10 :                      GLw1.bt10_bt11_HBeam0x349e460 : GLw1.bt10_bt11_HBeam_phys0x349e5e0 GLw1.bt10_bt11_HBeam_phys0x349e6b0 GLw1.bt10_bt11_HBeam_phys0x349e7b0 
            10 :                      GLw1.up10_up11_HBeam0x3465cb0 : GLw1.up10_up11_HBeam_phys0x3465e30 GLw1.up10_up11_HBeam_phys0x3465f00 GLw1.up10_up11_HBeam_phys0x3466000 
             8 :                               lSJReceiver0x364d2f0 : lSJReceiver_phys0x364d430 lSJReceiver_phys0x364d530 lSJReceiver_phys0x364d630 
             2 :                              lSJCLSanchor0x3649140 : lSJCLSanchor_phys0x36492c0 lSJCLSanchor_phys0x36493c0 
             1 :                                  lAcrylic0x34fe480 : pAcrylic0x34fed00 
             1 :                                    lAirTT0x4ee6b70 : pTopTracker0x4ef4a10 
             1 :                                  lBtmRock0x3464aa0 : pBtmRock0x34652e0 
             1 :                                  lExpHall0x3460090 : pExpHall0x3460190 
             1 :                               lInnerWater0x34fde10 : pInnerWater0x34fec60 
             1 :                             lLowerChimney0x4ee4270 : lLowerChimney_phys0x4ee5e60 
             1 :                      lLowerChimneyAcrylic0x4ee4490 : pLowerChimneyAcrylic0x4ee49d0 
             1 :                           lLowerChimneyLS0x4ee46a0 : pLowerChimneyLS0x4ee4a90 
             1 :                        lLowerChimneySteel0x4ee48c0 : pLowerChimneySteel0x4ee4b60 
             1 :                           lOuterWaterPool0x3465550 : pOuterWaterPool0x34fd080 
             1 :                               lPoolLining0x3465180 : pPoolLining0x3465240 
             1 :                            lReflectorInCD0x34fd7a0 : pCentralDetector0x34fee50 
             1 :                                   lTarget0x34feaf0 : pTarget0x34feda0 
             1 :                                  lTopRock0x345fc10 : pTopRock0x345fce0 
             1 :                             lUpperChimney0x4ee1f50 : lUpperChimney_phys0x4ee39c0 
             1 :                           lUpperChimneyLS0x4ee2050 : pUpperChimneyLS0x4ee2370 
             1 :                        lUpperChimneySteel0x4ee2160 : pUpperChimneySteel0x4ee2440 
             1 :                        lUpperChimneyTyvek0x4ee2270 : pUpperChimneyTyvek0x4ee2510 
             1 :                                    lWorld0x344f8d0 : lWorld0x344f8d0_PV 
    slice 0:10:1 
    []




GItemList/GMeshLib.txt solid names for each lvIdx::

    090 sTarget0x34fe8a0
     91 sAcrylic0x34fe230
     92 sStrut0x3501680
     93 sStrut0x3559670                         

     94 sStrutBallhead0x352a360                                     6 pts Y  GPts.NumPt     1 lvIdx ( 93)
     95 uni10x34cdcb0                                               7 pts Y  GPts.NumPt     1 lvIdx ( 94) 
     96 base_steel0x360d8f0                                         8 pts Y  GPts.NumPt     1 lvIdx ( 95) 
     97 uni_acrylic30x35ff3d0                                      **5 pts Y  GPts.NumPt     1 lvIdx ( 96)**

     98 solidXJanchor0x363f2f0

     99 NNVTMCPPMTsMask0x3c9fa80                                    2 pts Y  GPts.NumPt     6 lvIdx ( 103 98 102 101 99 100)
    100 NNVTMCPPMT_PMT_20inch_inner1_solid_1_Ellipsoid0x3503950
    101 NNVTMCPPMT_PMT_20inch_inner2_solid0x3cae8f0
    102 NNVTMCPPMT_PMT_20inch_body_solid0x3cad240
    103 NNVTMCPPMT_PMT_20inch_pmt_solid0x3ca9320
    104 NNVTMCPPMTsMask_virtual0x3cb3b40

    105 HamamatsuR12860sMask0x3c9afa0                                3 pts Y  GPts.NumPt     6 lvIdx ( 109 104 108 107 105 106)
    106 HamamatsuR12860_PMT_20inch_inner1_solid_I0x3c96fa0
    107 HamamatsuR12860_PMT_20inch_inner2_solid_1_90x3c93610
    108 HamamatsuR12860_PMT_20inch_body_solid_1_90x3ca7680
    109 HamamatsuR12860_PMT_20inch_pmt_solid_1_90x3cb68e0
    110 HamamatsuR12860sMask_virtual0x3c99fb0

    111 PMT_3inch_inner1_solid_ell_helper0x4436560                  1 pts Y  GPts.NumPt     5 lvIdx ( 114 112 110 111 113)
    112 PMT_3inch_inner2_solid_ell_helper0x4436640
    113 PMT_3inch_body_solid_ell_ell_helper0x44364d0
    114 PMT_3inch_cntr_solid0x44366d0
    115 PMT_3inch_pmt_solid0x4436210

    116 sChimneyAcrylic0x4ee4370


    120 sInnerWater0x34fdbc0
    121 sReflectorInCD0x34fd550


    122 mask_PMT_20inch_vetosMask0x3ca1aa0                         4 pts Y  GPts.NumPt     6 lvIdx ( 126 121 125 124 122 123)
    123 PMT_20inch_veto_inner1_solid0x3ca4f10
    124 PMT_20inch_veto_inner2_solid0x3ca5130
    125 PMT_20inch_veto_body_solid_1_20x3ca4230
    126 PMT_20inch_veto_pmt_solid_1_20x3ca38b0
    127 mask_PMT_20inch_vetosMask_virtual0x3ca0a80

    128 sOuterWaterPool0x3465440
    129 sPoolLining0x3464b60





PROBLEM MM 5 (CAUTION UNCONTROLLED MM INDEX IN 5/6/7/8) lvIdx 96  
-------------------------------------------------------------------- 

::

    2021-04-19 02:35:44.248 INFO  [32586] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2021-04-19 02:35:44.248 INFO  [32586] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0xbef4c0
    mm index   0 geocode   A                  numVolumes       3084 numFaces      183096 numITransforms           1 numITransforms*numVolumes        3084 GParts Y GPts Y
    mm index   1 geocode   A                  numVolumes          5 numFaces        1584 numITransforms       25600 numITransforms*numVolumes      128000 GParts Y GPts Y
    mm index   2 geocode   A                  numVolumes          6 numFaces        3504 numITransforms       12612 numITransforms*numVolumes       75672 GParts Y GPts Y
    mm index   3 geocode   A                  numVolumes          6 numFaces        5980 numITransforms        5000 numITransforms*numVolumes       30000 GParts Y GPts Y
    mm index   4 geocode   A                  numVolumes          6 numFaces        3284 numITransforms        2400 numITransforms*numVolumes       14400 GParts Y GPts Y

    mm index   5 geocode   A                  numVolumes          1 numFaces        1272 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   6 geocode   A                  numVolumes          1 numFaces         528 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   7 geocode   A                  numVolumes          1 numFaces         960 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   8 geocode   A                  numVolumes          1 numFaces         384 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y

    mm index   9 geocode   A                  numVolumes        130 numFaces        1560 numITransforms         504 numITransforms*numVolumes       65520 GParts Y GPts Y
     num_remainder_volumes 3084 num_instanced_volumes 315952 num_remainder_volumes + num_instanced_volumes 319036 num_total_faces 202152 num_total_faces_woi 125348744 (woi:without instancing) 
       0 pts Y  GPts.NumPt  3084 lvIdx ( 130 12 11 3 0 1 2 10 9 8 ... 88 88 88 88 88 118 115 116 117)

       1 pts Y  GPts.NumPt     5 lvIdx ( 114 112 110 111 113)
       2 pts Y  GPts.NumPt     6 lvIdx ( 103 98 102 101 99 100)
       3 pts Y  GPts.NumPt     6 lvIdx ( 109 104 108 107 105 106)
       4 pts Y  GPts.NumPt     6 lvIdx ( 126 121 125 124 122 123)

     **5 pts Y  GPts.NumPt     1 lvIdx ( 96)**

       6 pts Y  GPts.NumPt     1 lvIdx ( 93)
       7 pts Y  GPts.NumPt     1 lvIdx ( 94)
       8 pts Y  GPts.NumPt     1 lvIdx ( 95)


       9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)


    2021-04-19 02:35:44.249 INFO  [32586] [OGeo::convert@301] [ nmm 10
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
    2021-04-19 02:35:44.278 ERROR [32586] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
    2021-04-19 02:35:44.278 ERROR [32586] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
    2021-04-19 02:35:44.279 ERROR [32586] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
    2021-04-19 02:35:44.279 ERROR [32586] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
    2021-04-19 02:35:44.279 INFO  [32586] [OGeo::convert@322] ] nmm 10
    2021-04-19 02:35:44.280 INFO  [32586] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
    2021-04-19 02:35:44.280 INFO  [32586] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix snap-emm-5-
    2021-04-19 02:35:44.280 ERROR [32586] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn 304632 cmdline_target 0 gdmlaux_target -1 active_target 304632
    2021-04-19 02:35:44.281 INFO  [32586] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
     count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/snap-emm-5-00000.jpg dt     1.1119
    2021-04-19 02:35:45.546 INFO  [32586] [OTracer::report@192] OpTracer::snap
     trace_count              1 trace_prep          0.0005 avg     0.0005
     trace_time          1.1774 avg     1.1774

    2021-04-19 02:35:45.547 INFO  [32586] [BTimes::dump@183] OTracer::report
                  validate000                   0.0003
                   compile000                   0.0000
                 prelaunch000                   0.0639
                    launch000                   1.1119
                    launchAVG                   1.1119
    2021-04-19 02:35:45.547 INFO  [32586] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210419_023542
    2021-04-19 02:35:45.547 INFO  [32586] [BFile::preparePath@844] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210419_023542
    2021-04-19 02:35:45.548 INFO  [32586] [OpTracer::snap@182] ]





Issue: 2021 April : new geometry timings much lower ? Whats causing the slowdown ?
--------------------------------------------------------------------------------------

::

    OpSnapTest --xanalytic --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 


::

    O[blyth@localhost opticks]$ UseOptiX --uniqrec
    TITAN_V/0
    TITAN_RTX/1


* is --xanalytic still needed ?
* --enabledmergedmesh seems not working ?

::

    OpSnapTest --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 --enabledmergedmesh 1


    2021-04-17 02:39:48.003 INFO  [157145] [BTimes::dump@183] OTracer::report
                  validate000                   0.0251
                   compile000                   0.0000
                 prelaunch000                   1.2260
                    launch000                   0.0023
                    launchAVG                   0.0023
    2021-04-17 02:39:48.003 INFO  [157145] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_023944
    2021-04-17 02:39:48.003 INFO  [157145] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_023944
    2021-04-17 02:39:48.004 INFO  [157145] [OpTracer::snap@180] ]
    O[blyth@localhost optixrap]$ 



geocache-simple-mm(){ ls -1 $(geocache-keydir)/GMergedMesh ; }
geocache-simple()
{
    local mm
    local cmd 
    for mm in $(geocache-simple-mm) ; do 
        cmd="OpSnapTest --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 --enabledmergedmesh $mm"
        echo $cmd
    done 
}


Suspect the problem will be the "temple"  : NOPE THE TEMPLE NOT
--------------------------------------------------------------------

* warning the "5/" is before pinning down repeat_candidate ordering with the two-level-sort 


::

    O[blyth@localhost opticks]$ python3 ana/ggeo.py 5/
    nidx:70258 triplet:5000000 sh:600010 sidx:    0   nrpo( 70258     5     0     0 )  shape(  96  16                              uni_acrylic3                          Water///Acrylic) 

    gt : gg.all_volume_transforms[70258]
    [[   -0.585    -0.805     0.098     0.   ]
     [   -0.809     0.588     0.        0.   ]
     [   -0.057    -0.079    -0.995     0.   ]
     [ 1022.116  1406.822 17734.953     1.   ]]

    tr : transform
    [[   -0.585    -0.805     0.098     0.   ]
     [   -0.809     0.588     0.        0.   ]
     [   -0.057    -0.079    -0.995     0.   ]
     [ 1022.116  1406.822 17734.953     1.   ]]

    it : inverted transform
    [[   -0.585    -0.809    -0.057     0.   ]
     [   -0.805     0.588    -0.079     0.   ]
     [    0.098    -0.       -0.995     0.   ]
     [   -0.       -0.    17820.        1.   ]]

    bb : bbox4
    [[  574.885   960.342 17685.367     1.   ]
     [ 1469.02   1852.852 17893.8       1.   ]]

    cbb : (bb[0]+bb[1])/2.
    [ 1021.952  1406.597 17789.584     1.   ]

    c4 : center4
    [ 1021.952  1406.597 17789.584     1.   ]

    ce : center_extent
    [ 1021.952  1406.597 17789.584   447.067]

    ic4 : np.dot( c4, it) : inverse transform applied to center4 : expect close to origin 
    [  5.608  -0.    -54.344   1.   ]

    ibb : np.dot( bb, it) : inverse transform applied to bbox4 : expect symmetric around origin
    [[ 616.268   99.383  110.248    1.   ]
     [-605.053  -99.383 -218.936    1.   ]]





geocache-simple
---------------------

::

    O[blyth@localhost opticks]$ geocache-simple()
    > {
    >     local mm
    >     local cmd 
    >     for mm in $(geocache-simple-mm) ; do   
    >         cmd="OpSnapTest --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 --enabledmergedmesh $mm --snapoverrideprefix simple-enabledmergedmesh-$mm"
    >         echo $cmd
    >         eval $cmd 
    >     done 
    > }



