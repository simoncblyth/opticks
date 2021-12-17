cxs_intersect_meshname_key_clearly_wrong
===========================================


workstation::

     cx
     ./cxs.sh 

laptop:: 

     cx
     ./cxs_grab.sh 
     ./cxs.sh 


Fixed : the vnames need to use absolute indices unlike uval, ucount
-------------------------------------------------------------------------

* BUT the legend covers most of the plot

::

    positions_plt feat.name pid 
    INFO:__main__: xlim[0] -422.4000 xlim[1] 422.4000 
    INFO:__main__: ylim[0]   0.0000 ylim[1]   0.0000 
    INFO:__main__: zlim[0] -237.5996 zlim[1] 237.5996 
    /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1/CSG_GGeo/CSGOptiXSimulateTest/Hama_1/figs/positions_mpplt_pid.png
    positions_pvplt feat.name pid 
      0 : 3096 :  79004 :                  red :             HamamatsuR12860sMask_virtual : HamamatsuR12860sMask_virtual 
      1 : 3101 :  59971 :                green : HamamatsuR12860_PMT_20inch_inner1_solid_I : HamamatsuR12860_PMT_20inch_inner1_s_I 
      2 : 3102 :  40086 :                 blue : HamamatsuR12860_PMT_20inch_inner2_solid_1_4 : HamamatsuR12860_PMT_20inch_inner2_s_1_4 
      3 : 3097 :  33246 :                 cyan :                     HamamatsuR12860sMask : HamamatsuR12860sMask 
      4 : 2321 :  25876 :              magenta :                              sInnerWater : sInnerWater 
      5 : 3098 :  22118 :               yellow :                      HamamatsuR12860Tail : HamamatsuR12860Tail 
      6 : 2322 :  19394 :                 pink :                                 sAcrylic : sAcrylic 
      7 : 3089 :  14446 :         antiquewhite :                  NNVTMCPPMTsMask_virtual : NNVTMCPPMTsMask_virtual 
      8 : 3099 :   4956 :                 aqua : HamamatsuR12860_PMT_20inch_pmt_solid_1_4 : HamamatsuR12860_PMT_20inch_pmt_s_1_4 
      9 : 3110 :   4540 :           aquamarine :                             uni_acrylic3 : uni_acrylic3 
     10 : 3100 :   3057 :                azure : HamamatsuR12860_PMT_20inch_body_solid_1_4 : HamamatsuR12860_PMT_20inch_body_s_1_4 
     11 : 2320 :   2676 :                beige :                           sReflectorInCD : sReflectorInCD 
     12 : 3103 :   1040 :               bisque :                mask_PMT_20inch_vetosMask : mask_PMT_20inch_vetosMask 
     13 : 2447 :    813 :                black :                                   sStrut : sStrut 
     14 : 2433 :    804 :       blanchedalmond :                                   sStrut : sStrut 
     15 :  197 :    452 :           blueviolet :                              sBottomRock : sBottomRock 
     16 :  837 :    349 :                brown :           GLb3.up11_FlangeI_Web_FlangeII : GLb3.up11_FlangeI_Web_FlangeII 
     17 :  848 :    148 :            burlywood :           GLb3.up11_FlangeI_Web_FlangeII : GLb3.up11_FlangeI_Web_FlangeII 
     18 :  205 :    128 :            cadetblue :      GLw1.up10_up11_FlangeI_Web_FlangeII : GLw1.up10_up11_FlangeI_Web_FlangeII 
     19 : 3109 :    116 :           chartreuse :                               base_steel : base_steel 
     20 :  866 :     75 :            chocolate :           GLb4.up10_FlangeI_Web_FlangeII : GLb4.up10_FlangeI_Web_FlangeII 
     21 : 3091 :     68 :                coral :                           NNVTMCPPMTTail : NNVTMCPPMTTail 
     22 :  209 :     52 :       cornflowerblue :      GLw1.up10_up11_FlangeI_Web_FlangeII : GLw1.up10_up11_FlangeI_Web_FlangeII 
     23 :  896 :     49 :             cornsilk :           GLb3.up09_FlangeI_Web_FlangeII : GLb3.up09_FlangeI_Web_FlangeII 
     24 :  879 :     21 :              crimson :           GLb4.up10_FlangeI_Web_FlangeII : GLb4.up10_FlangeI_Web_FlangeII 
     25 : 3107 :     15 :             darkblue :                           sStrutBallhead : sStrutBallhead 




Issue
-------

Whilst viewing a cross section thru PMT get names of totally different pieces of geometry in the mpl key legend and in output::

    INFO:__main__: zlim[0] -237.5996 zlim[1] 237.5996 
    /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1/CSG_GGeo/CSGOptiXSimulateTest/Hama_1/figs/positions_mpplt_pid.png
    positions_pvplt feat.name pid 
      0 : 3096 :  79004 :                  red :                                   sPlane : sPlane 
      1 : 3101 :  59971 :                green :                                    sWall : sWall 
      2 : 3102 :  40086 :                 blue :                                   sPlane : sPlane 
      3 : 3097 :  33246 :                 cyan :                                   sPlane : sPlane 
      4 : 2321 :  25876 :              magenta :                                   sPlane : sPlane 
      5 : 3098 :  22118 :               yellow :                                    sWall : sWall 
      6 : 2322 :  19394 :                 pink :                                   sPlane : sPlane 
      7 : 3089 :  14446 :         antiquewhite :                                   sPlane : sPlane 
      8 : 3099 :   4956 :                 aqua :                                   sPlane : sPlane 
      9 : 3110 :   4540 :           aquamarine :                                   sPlane : sPlane 
     10 : 3100 :   3057 :                azure :                                   sPlane : sPlane 
     11 : 2320 :   2676 :                beige :                                    sWall : sWall 
     12 : 3103 :   1040 :               bisque :                                   sPlane : sPlane 
     13 : 2447 :    813 :                black :                                   sPlane : sPlane 
     14 : 2433 :    804 :       blanchedalmond :                                    sWall : sWall 
     15 :  197 :    452 :           blueviolet :                                   sWorld : sWorld 
     16 :  837 :    349 :                brown :                            Upper_Chimney : Upper_Chimney 
     17 :  848 :    148 :            burlywood :                            Upper_LS_tube : Upper_LS_tube 
     18 :  205 :    128 :            cadetblue :                                 sTopRock : sTopRock 
     19 : 3109 :    116 :           chartreuse :                                   sPlane : sPlane 
     20 :  866 :     75 :            chocolate :                         Upper_Steel_tube : Upper_Steel_tube 
     21 : 3091 :     68 :                coral :                                    sWall : sWall 
     22 :  209 :     52 :       cornflowerblue :                                 sExpHall : sExpHall 
     23 :  896 :     49 :             cornsilk :                                   sAirTT : sAirTT 
     24 :  879 :     21 :              crimson :                         Upper_Tyvek_tube : Upper_Tyvek_tube 
     25 : 3107 :     15 :             darkblue :                                    sWall : sWall 
    /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1/CSG_GGeo/CSGOptiXSimulateTest/Hama_1/figs/positions_pvplt_pid.png


::

    In [1]: ph.pidfeat                                                                                                                                                                                        
    Out[1]: 
    Feature name pid val (313500,)
    uval [ 197  205  209  837  848  866  879  896 2320 2321 2322 2433 2447 3089 3091 3096 3097 3098 3099 3100 3101 3102 3103 3107 3109 3110] 
    ucount [  452   128    52   349   148    75    21    49  2676 25876 19394   804   813 14446    68 79004 33246 22118  4956  3057 59971 40086  1040    15   116  4540] 
    idxdesc [15 20 21 16  9 17 10 13 18 25 19  8 22 12 11  0  3  4  1 24  5 14  2  7  6 23] 
    onames sPlane sWall sPlane sPlane sPlane sWall sPlane sPlane sPlane sPlane sPlane sWall sPlane sPlane sWall sWorld Upper_Chimney Upper_LS_tube sTopRock sPlane Upper_Steel_tube sWall sExpHall sAirTT Upper_Tyvek_tube sWall 
    ocount [79004, 59971, 40086, 33246, 25876, 22118, 19394, 14446, 4956, 4540, 3057, 2676, 1040, 813, 804, 452, 349, 148, 128, 116, 75, 68, 52, 49, 21, 15] 
    ouval 3096 3101 3102 3097 2321 3098 2322 3089 3099 3110 3100 2320 3103 2447 2433 197 837 848 205 3109 866 3091 209 896 879 3107 



Could be a problem of the contents of key legend being based on frequencies of all intersects but only a small portion is in view at once. 

* TODO: Try some selective plotting to check the pids. 


::


    In [10]: ins = cxs.photons.view(np.int32)[:,3,3] & 0xffff                                                                                                                                                 

    In [11]: pid = cxs.photons.view(np.int32)[:,3,3] >> 16                                                                                                                                                    

    In [12]: pid                                                                                                                                                                                              
    Out[12]: array([2321, 2321, 3096, 2321, 2321, ..., 3096, 2433, 2322, 3096, 2433], dtype=int32)

    In [13]: ins                                                                                                                                                                                              
    Out[13]: array([    0,     0, 38213,     0,     0, ..., 38215,     0,     0, 38213,     0], dtype=int32)

    In [14]: np.unique(ins)                                                                                                                                                                                   
    Out[14]: 
    array([    0, 25602, 25609, 25631, 25639, 25651, 25676, 25688, 25705, 25754, 25795, 25818, 25864, 25901, 26010, 26112, 26143, 38213, 38215, 38219, 38221, 38225, 38228, 38237, 38260, 38267, 38282,
           38307, 38341, 38352, 38363, 38386, 38441, 38455, 43213, 43214, 43217, 43240, 45668, 45682, 46848, 46862, 47385, 47387, 47396, 47402, 47413, 47422, 47438, 47452], dtype=int32)

    In [15]: np.unique(ins, return_counts=True)                                                                                                                                                               
    Out[15]: 
    (array([    0, 25602, 25609, 25631, 25639, 25651, 25676, 25688, 25705, 25754, 25795, 25818, 25864, 25901, 26010, 26112, 26143, 38213, 38215, 38219, 38221, 38225, 38228, 38237, 38260, 38267, 38282,
            38307, 38341, 38352, 38363, 38386, 38441, 38455, 43213, 43214, 43217, 43240, 45668, 45682, 46848, 46862, 47385, 47387, 47396, 47402, 47413, 47422, 47438, 47452], dtype=int32),
     array([ 50837,  13081,    147,     29,     35,    149,    103,     66,    103,     74,    109,     58,     75,     89,    136,    135,    125, 195436,  45540,    314,    153,     89,     53,    131,
                93,     83,     87,     87,    167,      8,     16,    154,     16,     11,    369,    378,    197,     96,     10,      5,     71,     45,   2060,   1368,    413,    312,    122,    163,
                97,      5]))

    In [16]:                                           



