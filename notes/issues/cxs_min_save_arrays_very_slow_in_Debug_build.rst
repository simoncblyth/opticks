cxs_min_save_arrays_very_slow_in_Debug_build
==============================================


RESOLVED : Issue of flaky write speed of 1.2G hit.npy
------------------------------------------------------

* HMM : NOW CAN HARDLY SEE THE GAPS BETWEEN EVT IN nvtop
* found that sometimes 1.2G hit.npy took 10s to write, sometimes 0.5s
* turns out to be problem of TMP using /data1 HDD (from VIP_LOCAL_BASE)
* instead move TMP to use /tmp from VIP_LOCAL_TEMP
* Debug vs Release difference was a red herring - the issue occurs in both builds



Scan with writing to /tmp
---------------------------

::

    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (81, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                       init 1793256188.878504 1793256188.885516              7012                 0                 2          0.007012                              init 
                                  load_geom 1793256188.885516 1793256190.476812           1591296                 2                 3          1.591296                         load_geom 
                                upload_geom 1793256190.476843 1793256190.715749            238906                 4                 5          0.238906                       upload_geom 
                              slice_genstep 1793256190.715992 1793256190.716016                24                 9                10          0.000024                     slice_genstep 
                       upload genstep slice 1793256190.716022 1793256190.738580             22558                11                12          0.022558              upload genstep slice 
    ( 0)                     simulate slice 1793256190.738580 1793256205.598696          14860116                12                13         14.860116                    simulate slice 
                             download slice 1793256205.598696 1793256205.958802            360106                13                14          0.360106                    download slice 
                             concat slices  1793256205.958858 1793256205.958875                17                15                16          0.000017                    concat slices  
                               save arrays  1793256205.958891 1793256206.363819            404928                17                33          0.404928                      save arrays  
                              slice_genstep 1793256206.363961 1793256206.363972                11                35                36          0.000011                     slice_genstep 
                       upload genstep slice 1793256206.363977 1793256206.366544              2567                37                38          0.002567              upload genstep slice 
    ( 1)                     simulate slice 1793256206.366544 1793256220.566804          14200260                38                39         14.200260                    simulate slice 
                             download slice 1793256220.566804 1793256220.929125            362321                39                40          0.362321                    download slice 
                             concat slices  1793256220.929172 1793256220.929188                16                41                42          0.000016                    concat slices  
                               save arrays  1793256220.929199 1793256221.327660            398461                43                59          0.398461                      save arrays  
                              slice_genstep 1793256221.327755 1793256221.327767                12                61                62          0.000012                     slice_genstep 
                       upload genstep slice 1793256221.327773 1793256221.330370              2597                63                64          0.002597              upload genstep slice 
    ( 2)                     simulate slice 1793256221.330370 1793256235.192648          13862278                64                65         13.862278                    simulate slice 
                             download slice 1793256235.192648 1793256235.554623            361975                65                66          0.361975                    download slice 
                             concat slices  1793256235.554667 1793256235.554682                15                67                68          0.000015                    concat slices  
                               save arrays  1793256235.554693 1793256235.953148            398455                69                85          0.398455                      save arrays  
                              slice_genstep 1793256235.953276 1793256235.953290                14                87                88          0.000014                     slice_genstep 
                       upload genstep slice 1793256235.953296 1793256235.955905              2609                89                90          0.002609              upload genstep slice 
    ( 3)                     simulate slice 1793256235.955905 1793256249.884839          13928934                90                91         13.928934                    simulate slice 
                             download slice 1793256249.884839 1793256250.245726            360887                91                92          0.360887                    download slice 
                             concat slices  1793256250.245770 1793256250.245786                16                93                94          0.000016                    concat slices  
                               save arrays  1793256250.245795 1793256250.632924            387129                95               111          0.387129                      save arrays  
                              slice_genstep 1793256250.633063 1793256250.633107                44               113               114          0.000044                     slice_genstep 
                       upload genstep slice 1793256250.633113 1793256250.635686              2573               115               116          0.002573              upload genstep slice 
    ( 4)                     simulate slice 1793256250.635686 1793256264.441936          13806250               116               117         13.806250                    simulate slice 
                             download slice 1793256264.441936 1793256264.802542            360606               117               118          0.360606                    download slice 
                             concat slices  1793256264.802599 1793256264.802615                16               119               120          0.000016                    concat slices  
                               save arrays  1793256264.802626 1793256265.188561            385935               121               137          0.385935                      save arrays  
                              slice_genstep 1793256265.188730 1793256265.188743                13               139               140          0.000013                     slice_genstep 
                       upload genstep slice 1793256265.188748 1793256265.191328              2580               141               142          0.002580              upload genstep slice 
    ( 5)                     simulate slice 1793256265.191328 1793256279.075544          13884216               142               143         13.884216                    simulate slice 
                             download slice 1793256279.075544 1793256279.436132            360588               143               144          0.360588                    download slice 
                             concat slices  1793256279.436184 1793256279.436198                14               145               146          0.000014                    concat slices  
                               save arrays  1793256279.436208 1793256279.823756            387548               147               163          0.387548                      save arrays  
                              slice_genstep 1793256279.823948 1793256279.823963                15               165               166          0.000015                     slice_genstep 
                       upload genstep slice 1793256279.823969 1793256279.826678              2709               167               168          0.002709              upload genstep slice 
    ( 6)                     simulate slice 1793256279.826678 1793256294.311323          14484645               168               169         14.484645                    simulate slice 
                             download slice 1793256294.311323 1793256294.671476            360153               169               170          0.360153                    download slice 
                             concat slices  1793256294.671538 1793256294.671553                15               171               172          0.000015                    concat slices  
                               save arrays  1793256294.671564 1793256295.065299            393735               173               189          0.393735                      save arrays  
                              slice_genstep 1793256295.065493 1793256295.065506                13               191               192          0.000013                     slice_genstep 
                       upload genstep slice 1793256295.065511 1793256295.068127              2616               193               194          0.002616              upload genstep slice 
    ( 7)                     simulate slice 1793256295.068127 1793256309.208134          14140007               194               195         14.140007                    simulate slice 
                             download slice 1793256309.208134 1793256309.568443            360309               195               196          0.360309                    download slice 
                             concat slices  1793256309.568493 1793256309.568509                16               197               198          0.000016                    concat slices  
                               save arrays  1793256309.568519 1793256309.976336            407817               199               215          0.407817                      save arrays  
                              slice_genstep 1793256309.976540 1793256309.976553                13               217               218          0.000013                     slice_genstep 
                       upload genstep slice 1793256309.976558 1793256309.979124              2566               219               220          0.002566              upload genstep slice 
    ( 8)                     simulate slice 1793256309.979124 1793256323.221342          13242218               220               221         13.242218                    simulate slice 
                             download slice 1793256323.221342 1793256323.636068            414726               221               222          0.414726                    download slice 
                             concat slices  1793256323.636127 1793256323.636142                15               223               224          0.000015                    concat slices  
                               save arrays  1793256323.636153 1793256324.030328            394175               225               241          0.394175                      save arrays  
                              slice_genstep 1793256324.030532 1793256324.030545                13               243               244          0.000013                     slice_genstep 
                       upload genstep slice 1793256324.030551 1793256324.033113              2562               245               246          0.002562              upload genstep slice 
    ( 9)                     simulate slice 1793256324.033113 1793256337.155998          13122885               246               247         13.122885                    simulate slice 
                             download slice 1793256337.155998 1793256337.573821            417823               247               248          0.417823                    download slice 
                             concat slices  1793256337.573867 1793256337.573881                14               249               250          0.000014                    concat slices  
                               save arrays  1793256337.573891 1793256337.985200            411309               251               267          0.411309                      save arrays  
                              slice_genstep 1793256337.985437 1793256337.985451                14               269               270          0.000014                     slice_genstep 
                       upload genstep slice 1793256337.985456 1793256337.988131              2675               271               272          0.002675              upload genstep slice 
    (10)                     simulate slice 1793256337.988131 1793256351.839899          13851768               272               273         13.851768                    simulate slice 
                             download slice 1793256351.839899 1793256352.202399            362500               273               274          0.362500                    download slice 
                             concat slices  1793256352.202445 1793256352.202460                15               275               276          0.000015                    concat slices  
                               save arrays  1793256352.202470 1793256352.595835            393365               277               293          0.393365                      save arrays  
                              slice_genstep 1793256352.596087 1793256352.596101                14               295               296          0.000014                     slice_genstep 
                       upload genstep slice 1793256352.596107 1793256352.598690              2583               297               298          0.002583              upload genstep slice 
    (11)                     simulate slice 1793256352.598690 1793256366.731403          14132713               298               299         14.132713                    simulate slice 
                             download slice 1793256366.731403 1793256367.093853            362450               299               300          0.362450                    download slice 
                             concat slices  1793256367.093900 1793256367.093915                15               301               302          0.000015                    concat slices  
                               save arrays  1793256367.093926 1793256367.480542            386616               303               319          0.386616                      save arrays  
                              slice_genstep 1793256367.480787 1793256367.480801                14               321               322          0.000014                     slice_genstep 
                       upload genstep slice 1793256367.480806 1793256367.483381              2575               323               324          0.002575              upload genstep slice 
    (12)                     simulate slice 1793256367.483381 1793256389.197544          21714163               324               325         21.714163                    simulate slice 
                             download slice 1793256389.197544 1793256389.558199            360655               325               326          0.360655                    download slice 
                             concat slices  1793256389.558248 1793256389.558261                13               327               328          0.000013                    concat slices  
                               save arrays  1793256389.558271 1793256389.948286            390015               329               346          0.390015                      save arrays  
    num_timestamp 162 auto-offset from tmin 1793256188878504 2026-10-29T14:43:08.878
                                TOTAL: 145253758968610192 145253759169676627         201066435             13175             13453

     0 :                ta : ta






Even after fixing the improper QEvt gs deletion - flakiness in "save arrays" continues : another bug ?
-------------------------------------------------------------------------------------------------------

::

    ( 7)                     simulate slice 1793185607.526168 1793185621.384829          13858661               194               195         13.858661                    simulate slice 
                             download slice 1793185621.384829 1793185621.745672            360843               195               196          0.360843                    download slice 
                             concat slices  1793185621.745721 1793185621.745737                16               197               198          0.000016                    concat slices  
                               save arrays  1793185621.745747 1793185632.367158          10621411               199               215         10.621411                      save arrays  
                              slice_genstep 1793185632.367345 1793185632.367359                14               217               218          0.000014                     slice_genstep 
                       upload genstep slice 1793185632.367365 1793185632.370171              2806               219               220          0.002806              upload genstep slice 
    ( 8)                     simulate slice 1793185632.370171 1793185645.364816          12994645               220               221         12.994645                    simulate slice 
                             download slice 1793185645.364816 1793185645.728904            364088               221               222          0.364088                    download slice 
                             concat slices  1793185645.728956 1793185645.728971                15               223               224          0.000015                    concat slices  
                               save arrays  1793185645.728981 1793185656.051939          10322958               225               241         10.322958                      save arrays    <<<<<
                              slice_genstep 1793185656.056412 1793185656.056424                12               243               244          0.000012                     slice_genstep 
                       upload genstep slice 1793185656.056430 1793185656.059128              2698               245               246          0.002698              upload genstep slice 
    ( 9)                     simulate slice 1793185656.059128 1793185668.941198          12882070               246               247         12.882070                    simulate slice 
                             download slice 1793185668.941198 1793185669.302819            361621               247               248          0.361621                    download slice 
                             concat slices  1793185669.302880 1793185669.302895                15               249               250          0.000015                    concat slices  
                               save arrays  1793185669.302906 1793185679.119732           9816826               251               267          9.816826                      save arrays   <<<<<<
                              slice_genstep 1793185679.119970 1793185679.119984                14               269               270          0.000014                     slice_genstep 
                       upload genstep slice 1793185679.119990 1793185679.122574              2584               271               272          0.002584              upload genstep slice 
    (10)                     simulate slice 1793185679.122574 1793185692.750976          13628402               272               273         13.628402                    simulate slice 
                             download slice 1793185692.750976 1793185693.111480            360504               273               274          0.360504                    download slice 
                             concat slices  1793185693.111529 1793185693.111545                16               275               276          0.000016                    concat slices  
                               save arrays  1793185693.111556 1793185703.120873          10009317               277               293         10.009317                      save arrays    <<<<<<
                              slice_genstep 1793185703.121077 1793185703.121093                16               295               296          0.000016                     slice_genstep 
                       upload genstep slice 1793185703.121099 1793185703.123675              2576               297               298          0.002576              upload genstep slice 
    (11)                     simulate slice 1793185703.123675 1793185716.965192          13841517               298               299         13.841517                    simulate slice 
                             download slice 1793185716.965192 1793185717.327098            361906               299               300          0.361906                    download slice 
                             concat slices  1793185717.327148 1793185717.327163                15               301               302          0.000015                    concat slices  
                               save arrays  1793185717.327174 1793185717.869286            542112               303               319          0.542112                      save arrays     <<<<<
                              slice_genstep 1793185717.869494 1793185717.869506                12               321               322          0.000012                     slice_genstep 
                       upload genstep slice 1793185717.869512 1793185717.872091              2579               323               324          0.002579              upload genstep slice 
    (12)                     simulate slice 1793185717.872091 1793185739.489037          21616946               324               325         21.616946                    simulate slice 
                             download slice 1793185739.489037 1793185739.855003            365966               325               326          0.365966                    download slice 
                             concat slices  1793185739.855053 1793185739.855070                17               327               328          0.000017                    concat slices  
                               save arrays  1793185739.855087 1793185740.509495            654408               329               346          0.654408                      save arrays     <<<<
    num_timestamp 162 auto-offset from tmin 1793185435243291 2026-10-28T19:03:55.243
                                TOTAL: 145248032298844927 145248032604100795         305255868             13175             13453





::

     16.706617   0.000007 2026-10-29 09:53:46.895 16412.99  2481.19 A000_QSim__reset_HEAD
     16.706623   0.000006 2026-10-29 09:53:46.895 16412.99  2481.19 A000_SEvt__endOfEvent_HEAD
     16.706636   0.000013 2026-10-29 09:53:46.895 16412.99  2481.19 A000_SEvt__endIndex
     16.706677   0.000041 2026-10-29 09:53:46.895 16412.99  2481.19 A000_SEvt__endOfEvent_BSAV
     16.706687   0.000010 2026-10-29 09:53:46.895 16412.99  2481.19 A000_SEvt__save_HEAD
     16.706698   0.000011 2026-10-29 09:53:46.895 16412.99  2481.19 A000_SEvt__save_ACOP
     16.706703   0.000005 2026-10-29 09:53:46.895 16412.99  2481.19 A000_SEvt__save_BHIT
     16.706708   0.000005 2026-10-29 09:53:46.895 16412.99  2481.19 A000_SEvt__save_AHIT
     16.707137   0.000429 2026-10-29 09:53:46.896 16412.99  2481.19 A000_SEvt__save_BSAV
     27.621330  10.914193 2026-10-29 09:53:57.810 16412.99  2481.19 A000_SEvt__save_ASAV
     27.649180   0.027850 2026-10-29 09:53:57.838 16412.99  2481.19 A000_SEvt__save_TAIL
     27.649210   0.000030 2026-10-29 09:53:57.838 16412.99  2481.19 A000_SEvt__endOfEvent_ASAV
     27.652651   0.003441 2026-10-29 09:53:57.841 15086.61  1155.23 A000_SEvt__endOfEvent_ACLE
     27.652738   0.000087 2026-10-29 09:53:57.841 15086.61  1155.23 A000_SEvt__EndOfRun
     27.652788   0.000050 2026-10-29 09:53:57.841 15086.61  1155.23 A000_SEvt__endOfEvent_TAIL


Time is all in NPFold::save::

    4697         SProf::Add("SEvt__save_BSAV");
    4698         save_fold->save(dir);
    4699         SProf::Add("SEvt__save_ASAV");


::

    /data1/blyth/local/opticks_Release_g411/bin/cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2026-10-29 10:24:30.902  902522735 : [/data1/blyth/local/opticks_Release_g411/bin/cxs_min.sh
    SGLM::initView VIEW [-] load_interpolated_view NO  interpolated_view.brief -
    2026-10-29 10:24:32.710 INFO  [2231079] [CSGOptiX::prepareParamSimulate_SER@1052] [Params_Helper::desc_SER hintMethod : 0 bounceInterval : 1 coherenceHintBits : -1]
    [NPFold__save_DUMP 2026-10-29 10:24:47.622 : [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000]
    [NPFold__save_DUMP 2026-10-29 10:24:47.622 : _save [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000]
    [NPFold__save_DUMP 2026-10-29 10:24:47.622 : _save_arrays [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000]
    -NPFold__save_DUMP 2026-10-29 10:24:47.622 : k [genstep.npy a [(10, 6, 4, )]
    -NPFold__save_DUMP 2026-10-29 10:24:47.622 : k [hit.npy a [(21731473, 4, 4, )]
    -NPFold__save_DUMP 2026-10-29 10:24:57.564 : k [kprofile.npy a [(64, 3, )]
    ]NPFold__save_DUMP 2026-10-29 10:24:57.592 : _save_arrays [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000]
    ]NPFold__save_DUMP 2026-10-29 10:24:57.592 : _save [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000]
    ]NPFold__save_DUMP 2026-10-29 10:24:57.592 : [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000]
    2026-10-29 10:24:57.703  703463593 : ]/data1/blyth/local/opticks_Release_g411/bin/cxs_min.sh
    /data1/blyth/local/opticks_Release_g411/bin/cxs_min.sh : LOGFILE CSGOptiXSMTest.log in LOGDIR /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt PWD /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt
    4 -rw-r--r--. 1 blyth blyth 168 Oct 29 10:24 CSGOptiXSMTest.log















Look into "save arrays" timing - seems flakey - Release running now back up to ~10s
-------------------------------------------------------------------------------------

* instrument SEvt::endOfEvent with more SProf::Add to see where the 10s is happening

* all from NPFold::save ?


::

    [lo] A[blyth@localhost ALL1_Release_Philox_medium_evt]$ REPORT=1 SProf.py 
      Delta(s)   Δprev(s) DateTime (+08)        VM(MB)  RSS(MB) Label / Comment
    --------------------------------------------------------------------------------------------
      0.000000   0.000000 2026-10-28 17:04:25.874    44.63     7.44 SEvt__Init_RUN_META
      0.007934   0.007934 2026-10-28 17:04:25.882    44.76     9.62 CSGOptiX__SimulateMain_HEAD
      0.007959   0.000025 2026-10-28 17:04:25.882    44.76     9.62 CSGFoundry__Load_HEAD
      1.547316   1.539357 2026-10-28 17:04:27.421  5281.86   836.27 CSGFoundry__Load_TAIL
      1.547348   0.000032 2026-10-28 17:04:27.421  5281.86   836.27 CSGOptiX__Create_HEAD
      1.810657   0.263309 2026-10-28 17:04:27.684  7919.23  1154.60 CSGOptiX__Create_TAIL
      1.810716   0.000059 2026-10-28 17:04:27.684  7919.23  1154.60 SEvt__BeginOfRun
      1.810839   0.000123 2026-10-28 17:04:27.685  7919.23  1154.60 SEvt__beginOfEvent_FIRST_EGPU
      1.810857   0.000018 2026-10-28 17:04:27.685  7919.23  1154.60 A000_SEvt__setIndex
      1.810924   0.000067 2026-10-28 17:04:27.685  7919.23  1154.60 A000_QSim__simulate_HEAD
      1.810946   0.000022 2026-10-28 17:04:27.685  7919.23  1154.60 A000_QSim__simulate_LBEG
      1.810951   0.000005 2026-10-28 17:04:27.685  7919.23  1154.60 A000_QSim__simulate_PRUP
      1.832827   0.021876 2026-10-28 17:04:27.707 15087.23  1157.66 A000_QSim__simulate_PREL
     16.497419  14.664592 2026-10-28 17:04:42.371 15087.23  1157.66 A000_QSim__simulate_POST
     16.861208   0.363789 2026-10-28 17:04:42.735 16413.62  2484.84 A000_QSim__simulate_DOWN
     16.861254   0.000046 2026-10-28 17:04:42.735 16413.62  2484.84 A000_QSim__simulate_LEND # slice=1,max_slot_M=100
     16.861273   0.000019 2026-10-28 17:04:42.735 16413.62  2484.84 A000_QSim__simulate_PCAT
     16.861290   0.000017 2026-10-28 17:04:42.735 16413.62  2484.84 A000_QSim__simulate_BRES # numGenstepCollected=10,numPhotonCollected=100000000,numHit=21731473
     16.861298   0.000008 2026-10-28 17:04:42.735 16413.62  2484.84 A000_QSim__reset_HEAD
     16.861308   0.000010 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__endOfEvent_HEAD
     16.861323   0.000015 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__endIndex
     16.861364   0.000041 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__endOfEvent_BSAV
     16.861376   0.000012 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__save_HEAD
     16.861388   0.000012 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__save_ACOP
     16.861394   0.000006 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__save_BHIT
     16.861400   0.000006 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__save_AHIT
     16.861520   0.000120 2026-10-28 17:04:42.735 16413.62  2484.84 A000_SEvt__save_BSAV
     27.213826  10.352306 2026-10-28 17:04:53.088 16413.62  2484.84 A000_SEvt__save_ASAV
     27.230744   0.016918 2026-10-28 17:04:53.104 16413.62  2484.84 A000_SEvt__save_TAIL
     27.230762   0.000018 2026-10-28 17:04:53.104 16413.62  2484.84 A000_SEvt__endOfEvent_ASAV
     27.233837   0.003075 2026-10-28 17:04:53.108 15087.23  1158.45 A000_SEvt__endOfEvent_ACLE
     27.233930   0.000093 2026-10-28 17:04:53.108 15087.23  1158.45 A000_SEvt__EndOfRun
     27.239723   0.005793 2026-10-28 17:04:53.113 15087.23  1158.45 A000_SEvt__endOfEvent_TAIL
     27.239730   0.000007 2026-10-28 17:04:53.113 15087.23  1158.45 A000_QSim__reset_TAIL
     27.239740   0.000010 2026-10-28 17:04:53.113 15087.23  1158.45 A000_QSim__simulate_TAIL
     27.240297   0.000557 2026-10-28 17:04:53.114 15087.23  1158.45 CSGOptiX__SimulateMain_TAIL
    Identity                                             Start(s)  Duration(s)     Sub PREL→POST(s)     Sub POST→DOWN(s)    Sub TAIL→RESET(s)    VM(MB)  RSS(MB)  Comment
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ALL1_Release_Philox_medium_evt/A000_QSim                1.811    15.050374            14.664592             0.363789           -10.378442   16413.62  2484.84  # numGenstepCollected=10,numPhotonCollected=100000000,numHit=21731473  # slice=1,max_slot_M=100


::

    2026-10-28 17:06:37.004 INFO  [2089064] [SEvt::save@4693] SEvt::descSaveDir dir_ $TMP/GEOM/$GEOM/$ExecutableName dir  /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000 reldir ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-none} SEventConfig::_EventReldirDefault ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-no_opticks_event_name} with_index Y index 0 this 0x1e0bd40

    2026-10-28 17:06:37.005 INFO  [2089064] [SEvt::save@4696] [ save_fold.save /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000
    2026-10-28 17:06:46.873 INFO  [2089064] [SEvt::save@4700] ] save_fold.save /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000
    2026-10-28 17:06:46.873 INFO  [2089064] [SEvt::saveFrame@4805] [ dir /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000
    2026-10-28 17:06:46.873 INFO  [2089064] [SEvt::saveFrame@4807] ] dir /data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000
    2026-10-28 17:06:46.877 INFO  [2089064] [SEvt::clear_output@2236] ]


Added NPFold_test::LoadSave - reveals a corrupt genstep::

    [NPFold__load_DUMP 2026-10-28 17:19:17.507 : [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000]
    terminate called after throwing an instance of 'std::bad_alloc'
      what():  std::bad_alloc

    Program received signal SIGABRT, Aborted.
    0x00007ffff788bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64
    (gdb) bt
    #0  0x00007ffff788bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff783eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff7828833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff7ca1b21 in __gnu_cxx::__verbose_terminate_handler() [clone .cold] () from /lib64/libstdc++.so.6
    #4  0x00007ffff7cad52c in __cxxabiv1::__terminate(void (*)()) () from /lib64/libstdc++.so.6
    #5  0x00007ffff7cad597 in std::terminate() () from /lib64/libstdc++.so.6
    #6  0x00007ffff7cad7f9 in __cxa_throw () from /lib64/libstdc++.so.6
    #7  0x00007ffff7ca1731 in operator new(unsigned long) [clone .cold] () from /lib64/libstdc++.so.6
    #8  0x000000000042f227 in __gnu_cxx::new_allocator<char>::allocate (this=0x464c40, __n=4807180017747461440) at /usr/include/c++/11/ext/new_allocator.h:127
    #9  0x000000000042a577 in std::allocator_traits<std::allocator<char> >::allocate (__a=..., __n=4807180017747461440) at /usr/include/c++/11/bits/alloc_traits.h:464
    #10 0x000000000042afee in std::_Vector_base<char, std::allocator<char> >::_M_allocate (this=0x464c40, __n=4807180017747461440) at /usr/include/c++/11/bits/stl_vector.h:346
    #11 0x0000000000426454 in std::vector<char, std::allocator<char> >::_M_default_append (this=0x464c40, __n=4807180017747461440) at /usr/include/c++/11/bits/vector.tcc:635
    #12 0x0000000000421e43 in std::vector<char, std::allocator<char> >::resize (this=0x464c40, __new_size=4807180017747461440) at /usr/include/c++/11/bits/stl_vector.h:940
    #13 0x000000000040cd08 in NP::decode_header (this=0x464c40, data_resize=true) at ../NP.hh:2254
    #14 0x0000000000415652 in NP::load (this=0x464c40, _path=0x464bc0 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000/genstep.npy", _sli=0x0) at ../NP.hh:9054
    #15 0x000000000040e170 in NP::Load_ (path=0x464bc0 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000/genstep.npy") at ../NP.hh:4651
    #16 0x000000000040dffc in NP::Load (path_=0x463b00 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000/genstep.npy") at ../NP.hh:4498
    #17 0x000000000040e2b6 in NP::Load (dir=0x463210 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000", name=0x463f40 "genstep.npy") at ../NP.hh:4703
    #18 0x000000000041ab90 in NPFold::load_array (this=0x463280, _base=0x463210 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000", relp=0x463f40 "genstep.npy") at ../NPFold.h:2711
    #19 0x000000000041b05b in NPFold::load_index (this=0x463280, _base=0x463210 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000") at ../NPFold.h:2876
    #20 0x000000000041b354 in NPFold::load (this=0x463280, _base=0x463210 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000") at ../NPFold.h:2917
    #21 0x0000000000417b37 in NPFold::Load_ (base=0x463210 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000") at ../NPFold.h:569
    #22 0x0000000000417c92 in NPFold::Load (base_=0x43b6c8 "/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Release_Philox_medium_evt/A000") at ../NPFold.h:607
    #23 0x0000000000404ae4 in NPFold_test::LoadSave () at NPFold_test.cc:28
    #24 0x0000000000404b09 in NPFold_test::Main () at NPFold_test.cc:36
    #25 0x0000000000404b14 in main () at NPFold_test.cc:42
    (gdb) 
    quit
    A debugging session is active.

        Inferior 1 [process 2090741] will be killed.


Smoking gun - the cxs_min.sh genstep has reproducible crazy shape and is header only, no data - from Release build, normal from Debug build::

    [lo] A[blyth@localhost A000]$ xxd genstep.npy 
    00000000: 934e 554d 5059 0100 a600 7b27 6465 7363  .NUMPY....{'desc
    00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2838 3532  e, 'shape': (852
    00000040: 3233 3039 3533 3831 3633 3939 3436 3839  2309538163994689
    00000050: 2c20 3739 3436 3939 3334 3432 3738 3138  , 79469934427818
    00000060: 3830 3138 302c 2037 3839 3032 3736 292c  80180, 7890276),
    00000070: 207d 2020 2020 2020 2020 2020 2020 2020   }              
    00000080: 2020 2020 2020 2020 2020 2020 2020 2020                  
    00000090: 2020 2020 2020 2020 2020 2020 2020 2020                  
    000000a0: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    [lo] A[blyth@localhost A000]$ 


    A[blyth@localhost A000]$ xxd genstep.npy
    00000000: 934e 554d 5059 0100 a600 7b27 6465 7363  .NUMPY....{'desc
    00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2838 3532  e, 'shape': (852
    00000040: 3233 3039 3533 3831 3633 3939 3436 3839  2309538163994689
    00000050: 2c20 3739 3436 3939 3334 3432 3738 3138  , 79469934427818
    00000060: 3830 3138 302c 2037 3839 3032 3736 292c  80180, 7890276),
    00000070: 207d 2020 2020 2020 2020 2020 2020 2020   }              
    00000080: 2020 2020 2020 2020 2020 2020 2020 2020                  
    00000090: 2020 2020 2020 2020 2020 2020 2020 2020                  
    000000a0: 2020 2020 2020 2020 2020 2020 2020 200a                 .



FOLD from Debug build has no problem - has reasonable load save times of fractions of a second::

    [NPFold__load_DUMP 2026-10-28 17:37:30.113 : [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt/A000]
    ]NPFold__load_DUMP 2026-10-28 17:37:30.545 : [/data1/blyth/tmp/GEOM/J26_4_1_opticks_Debug_g411/CSGOptiXSMTest/ALL1_Debug_Philox_medium_evt/A000 rc 0]
    [NPFold__save_DUMP 2026-10-28 17:37:30.545 : [/tmp/blyth/np/NPFold_test]
    ]NPFold__save_DUMP 2026-10-28 17:37:31.317 : [/tmp/blyth/np/NPFold_test]


Maybe : problem manifests as flakey slow save, but is actually a corrupted genstep ?

* NO : the Debug build is also slow saving - but genstep is OK







gs starts OK, then gets mangled in Release build::

    [lo] A[blyth@localhost CSGOptiX]$ ./cxs_min_bad_gs_debug.sh

    ...

    2026-10-28 18:09:18.179 INFO  [2106083] [QSim::simulate@495]    0 : sslice {       0,      10,         0, 100000000}100.000000
    2026-10-28 18:09:18.187 INFO  [2106083] [SEvt::setNumPhoton@2630]  evt->num_photon 100000000 evt->num_tag 0 evt->num_flat 0
    2026-10-28 18:09:18.202 INFO  [2106083] [CSGOptiX::prepareParamSimulate_SER@1052] [Params_Helper::desc_SER hintMethod : 0 bounceInterval : 1 coherenceHintBits : -1]
    2026-10-28 18:09:32.770 INFO  [2106083] [QSim::simulate@521]  eventID 0 xxl NO  i    0 dt   14.568136 slice    0 : sslice {       0,      10,         0, 100000000}100.000000
    2026-10-28 18:09:32.770 INFO  [2106083] [SEvt::gather_components@4068]  num_comp 3 from provider QEvt fkey f000
    2026-10-28 18:09:32.770 INFO  [2106083] [SEvt::gather_components@4078]  k         genstep a  <f4(10, 6, 4, ) null_component NO 
    2026-10-28 18:09:33.133 INFO  [2106083] [SEvt::gather_components@4078]  k             hit a  <f4(21731473, 4, 4, ) null_component NO 
    2026-10-28 18:09:33.133 INFO  [2106083] [SEvt::gather_components@4078]  k        kprofile a  <u8(64, 3, ) null_component NO 
    2026-10-28 18:09:33.133 INFO  [2106083] [SEvt::gather_components@4106]  num_comp 3 num_genstep 10 num_photon -1 num_hit 21731473 gather_total 1 genstep_total 10 photon_total 0 hit_total 21731473
    2026-10-28 18:09:33.133 INFO  [2106083] [QSim::simulate@554]  num_slice 1 has_hm NO  has_hlm NO  do_final_merge NO 
    2026-10-28 18:09:33.133 INFO  [2106083] [SEvt::gather_metadata@4151]  provmeta [NumPhotonCollected:100000000
    NumGenstepCollected:10
    MaxBounce:63
    site:SEvt::endMeta
    hitmask:8192
    index:0
    instance:0
    SEvt__beginOfEvent_0:1793182158179576,8108636,1180512
    SEvt__beginOfEvent_1:1793182158179819,8108636,1180512
    SEvt__endOfEvent_0:1793182173133499,16806888,2542316
    t_BeginOfEvent:1793182158179608
    t_PreLaunch:1793182158202482
    t_PostLaunch:1793182172770660
    t_EndOfEvent:1793182173133505
    t_Event:14953897
    t_Launch:14.5681
    ]
    2026-10-28 18:09:33.133 INFO  [2106083] [SEvt::save@4603] SEvt::DescComponent
     SEventConfig::DescGatherComp genstep,hit,kprofile

     SEventConfig::DescSaveComp genstep,hit,kprofile
                     hit   (21731473, 4, 4, ) 
                    seed                    - 
                 genstep (8522309538163994689, 7946993442781880180, 7890276, )       SEventConfig::MaxGenstep            10000000
                  photon                    -        SEventConfig::MaxPhoton       1000000000000
                  record                    -        SEventConfig::MaxRecord                   0
                     aux                    -           SEventConfig::MaxAux                   0
                     sup                    -           SEventConfig::MaxSup                   0
                     rec                    -           SEventConfig::MaxRec                   0
                     seq                    -           SEventConfig::MaxSeq                   0
                  domain                    - 
                simtrace                    - 
                 g4state                    - 
                     pho                    - 
                      gs                    - 

    2026-10-28 18:09:33.133 INFO  [2106083] [SEvt::save@4604] [NPFold::desc
    NPFold::desc_subfold
     tot_items 3
     folds 1


Debug build::

    2026-10-28 18:16:33.214 INFO  [2111453] [SEvt::gather_components@4078]  k        kprofile a  <u8(64, 3, ) null_component NO 
    2026-10-28 18:16:33.214 INFO  [2111453] [SEvt::gather_components@4106]  num_comp 3 num_genstep 10 num_photon -1 num_hit 21731510 gather_total 1 genstep_total 10 photon_total 0 hit_total 21731510
    2026-10-28 18:16:33.214 INFO  [2111453] [QSim::simulate@554]  num_slice 1 has_hm NO  has_hlm NO  do_final_merge NO 
    2026-10-28 18:16:33.214 INFO  [2111453] [SEvt::gather_metadata@4151]  provmeta [NumPhotonCollected:100000000
    NumGenstepCollected:10
    MaxBounce:63
    site:SEvt::endMeta
    hitmask:8192
    index:0
    instance:0
    SEvt__beginOfEvent_0:1793182570510398,8124068,1260184
    SEvt__beginOfEvent_1:1793182570510764,8124068,1260184
    SEvt__endOfEvent_0:1793182593214191,16822320,2624864
    t_BeginOfEvent:1793182570510439
    t_setGenstep_0:0
    t_setGenstep_1:0
    t_setGenstep_2:0
    t_setGenstep_3:1793182570510962
    t_setGenstep_4:1793182570511142
    t_setGenstep_5:1793182570511156
    t_setGenstep_6:1793182570511179
    t_setGenstep_7:1793182570513998
    t_setGenstep_8:1793182570526604
    t_PreLaunch:1793182570526644
    t_PostLaunch:1793182592850394
    t_EndOfEvent:1793182593214199
    t_Event:22703760
    t_Launch:22.3234
    ]
    2026-10-28 18:16:33.214 INFO  [2111453] [SEvt::save@4603] SEvt::DescComponent
     SEventConfig::DescGatherComp genstep,hit,kprofile

     SEventConfig::DescSaveComp genstep,hit,kprofile
                     hit   (21731510, 4, 4, ) 
                    seed                    - 
                 genstep         (10, 6, 4, )       SEventConfig::MaxGenstep            10000000
                  photon                    -        SEventConfig::MaxPhoton       1000000000000
                  record                    -        SEventConfig::MaxRecord                   0
                     aux                    -           SEventConfig::MaxAux                   0
                     sup                    -           SEventConfig::MaxSup                   0
                     rec                    -           SEventConfig::MaxRec                   0
                     seq                    -           SEventConfig::MaxSeq                   0
                  domain                    - 
                simtrace                    - 
                 g4state                    - 
                     pho                    - 
                      gs                    - 

    2026-10-28 18:16:33.214 INFO  [2111453] [SEvt::save@4604] [NPFold::desc
    NPFold::desc_subfold
     tot_items 3
     folds 1
     paths 1
      0 [/]  stamp:13



* the above DescComponent - just dumps what is seen in topfold

Under suspicion (as a bit complicated)::

    2220 void SEvt::clear_output()
    2221 {
    2222     setStage(SEvt__clear_output);
    2223 
    2224     LOG_IF(info, LIFECYCLE) << id() << " BEFORE clear_output_vector " ;
    2225 
    2226     clear_output_vector();
    2227 
    2228     const char* keylist = "genstep" ;
    2229     bool copy = false ;
    2230     char delim = ',' ;
    2231 
    2232     topfold->clear_except(keylist, copy, delim );
    2233 
    2234     LOG_IF(info, LIFECYCLE) << id() << " AFTER clear_output_vector " ;
    2235 
    2236     LOG(LEVEL) << "]" ;
    2237 }


Nope - thats done after the save.

Looks like the problem was calling QEvt::clear_gs too soon::

     876 void QSim::reset(int eventID)
     877 {
     878     SProf::Add("QSim__reset_HEAD");
     879     // qev->clear_gs();        // delete gs and set to nullptr - CANNOT BEFORE SEvt::endOfEvent AS CAUSES CORRUPTED SAVED GS 
     880     sev->endOfEvent(eventID);
     881     qev->clear_gs();
     882     LOG_IF(info, SEvt::LIFECYCLE) << "] eventID " << eventID ;
     883     SProf::Add("QSim__reset_TAIL");
     884 }
     885 



Actually - QEvt does not own the genstep (SEvt does) - so should not
be deleting it at all.


Somehow - the deletion failed to cause an issue in Debug build, but it did in Release.





Initial complaint
------------------

HUH compare back with Debug, notice that "save arrays" is factor of 10-20x slower for the Debug build ?

* from ~/o/notes/optixReorder.rst

SER scanning::

    TEST=ser_medium_scan cxs_min.sh

Compare Release build sreport::

    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792497633.035551 1792497633.043238              7687                 0                 2          0.007687                              init 
                             load_geom 1792497633.043238 1792497634.504547           1461309                 2                 3          1.461309                         load_geom 
                           upload_geom 1792497634.504803 1792497640.690812           6186009                 4                 5          6.186009                       upload_geom 
                         slice_genstep 1792497640.691098 1792497640.691122                24                 9                10          0.000024                     slice_genstep 
                  upload genstep slice 1792497640.691128 1792497640.713466             22338                11                12          0.022338              upload genstep slice 
                        simulate slice 1792497640.713466 1792497655.211095          14497629                12                13         14.497629                    simulate slice 
                        download slice 1792497655.211095 1792497655.576413            365318                13                14          0.365318                    download slice 
                        concat slices  1792497655.576469 1792497655.576489                20                15                16          0.000020                    concat slices  
                          save arrays  1792497655.576508 1792497656.154064            577556                17                21          0.577556                      save arrays  
                         slice_genstep 1792497656.154225 1792497656.154238                13                23                24          0.000013                     slice_genstep 
                  upload genstep slice 1792497656.154244 1792497656.156970              2726                25                26          0.002726              upload genstep slice 
                        simulate slice 1792497656.156970 1792497671.025224          14868254                26                27         14.868254                    simulate slice 
                        download slice 1792497671.025224 1792497671.388209            362985                27                28          0.362985                    download slice 
                        concat slices  1792497671.388272 1792497671.388288                16                29                30          0.000016                    concat slices  
                          save arrays  1792497671.388299 1792497672.032604            644305                31                35          0.644305                      save arrays  
                         slice_genstep 1792497672.058125 1792497672.058142                17                37                38          0.000017                     slice_genstep 
                  upload genstep slice 1792497672.058148 1792497672.060735              2587                39                40          0.002587              upload genstep slice 
                        simulate slice 1792497672.060735 1792497686.262012          14201277                40                41         14.201277                    simulate slice 
                        download slice 1792497686.262012 1792497686.623579            361567                41                42          0.361567                    download slice 
                        concat slices  1792497686.623642 1792497686.623658                16                43                44          0.000016                    concat slices  
                          save arrays  1792497686.623669 1792497687.268948            645279                45                49          0.645279                      save arrays  
                         slice_genstep 1792497687.273005 1792497687.273019                14                51                52          0.000014                     slice_genstep 
                  upload genstep slice 1792497687.273025 1792497687.275608              2583                53                54          0.002583              upload genstep slice 
                        simulate slice 1792497687.275608 1792497702.184684          14909076                54                55         14.909076                    simulate slice 
                        download slice 1792497702.184684 1792497702.611485            426801                55                56          0.426801                    download slice 
                        concat slices  1792497702.611539 1792497702.611555                16                57                58          0.000016                    concat slices  
                          save arrays  1792497702.611566 1792497703.252313            640747                59                63          0.640747                      save arrays  
                         slice_genstep 1792497703.280124 1792497703.280139                15                65                66          0.000015                     slice_genstep 
                  upload genstep slice 1792497703.280145 1792497703.282719              2574                67                68          0.002574              upload genstep slice 
                        simulate slice 1792497703.282719 1792497724.207686          20924967                68                69         20.924967                    simulate slice 
                        download slice 1792497724.207686 1792497724.629004            421318                69                70          0.421318                    download slice 
                        concat slices  1792497724.629050 1792497724.629065                15                71                72          0.000015                    concat slices  
                          save arrays  1792497724.629076 1792497725.246814            617738                73                78          0.617738                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792497633035551 2026-10-20T20:00:33.035
                                TOTAL: 59152423299535148 59152423391687944          92152796              1231              1281

     0 :                ta : ta
     1 :                tb : tb
     2 :                ab : ab


With Debug build::

    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792498487.262349 1792498487.272743             10394                 0                 2          0.010394                              init 
                             load_geom 1792498487.272743 1792498489.072099           1799356                 2                 3          1.799356                         load_geom 
                           upload_geom 1792498489.072135 1792498489.403941            331806                 4                 5          0.331806                       upload_geom 
                         slice_genstep 1792498489.404270 1792498489.404306                36                 9                10          0.000036                     slice_genstep 
                  upload genstep slice 1792498489.404312 1792498489.419749             15437                11                12          0.015437              upload genstep slice 
                        simulate slice 1792498489.419749 1792498511.858490          22438741                12                13         22.438741                    simulate slice 
                        download slice 1792498511.858490 1792498512.345051            486561                13                14          0.486561                    download slice 
                        concat slices  1792498512.345119 1792498512.345141                22                15                16          0.000022                    concat slices  
                          save arrays  1792498512.345165 1792498522.570141          10224976                17                21         10.224976                      save arrays  
                         slice_genstep 1792498522.570281 1792498522.570309                28                23                24          0.000028                     slice_genstep 
                  upload genstep slice 1792498522.570315 1792498522.572918              2603                25                26          0.002603              upload genstep slice 
                        simulate slice 1792498522.572918 1792498545.371080          22798162                26                27         22.798162                    simulate slice 
                        download slice 1792498545.371080 1792498545.845344            474264                27                28          0.474264                    download slice 
                        concat slices  1792498545.845403 1792498545.845423                20                29                30          0.000020                    concat slices  
                          save arrays  1792498545.845436 1792498555.597903           9752467                31                35          9.752467                      save arrays  
                         slice_genstep 1792498555.598044 1792498555.598072                28                37                38          0.000028                     slice_genstep 
                  upload genstep slice 1792498555.598087 1792498555.600683              2596                39                40          0.002596              upload genstep slice 
                        simulate slice 1792498555.600683 1792498577.763314          22162631                40                41         22.162631                    simulate slice 
                        download slice 1792498577.763314 1792498578.215727            452413                41                42          0.452413                    download slice 
                        concat slices  1792498578.215786 1792498578.215807                21                43                44          0.000021                    concat slices  
                          save arrays  1792498578.215820 1792498587.833406           9617586                45                49          9.617586                      save arrays  
                         slice_genstep 1792498587.838682 1792498587.838711                29                51                52          0.000029                     slice_genstep 
                  upload genstep slice 1792498587.838717 1792498587.841317              2600                53                54          0.002600              upload genstep slice 
                        simulate slice 1792498587.841317 1792498610.889796          23048479                54                55         23.048479                    simulate slice 
                        download slice 1792498610.889796 1792498611.332494            442698                55                56          0.442698                    download slice 
                        concat slices  1792498611.332547 1792498611.332568                21                57                58          0.000021                    concat slices  
                          save arrays  1792498611.332581 1792498620.965751           9633170                59                63          9.633170                      save arrays  
                         slice_genstep 1792498620.965914 1792498620.965943                29                65                66          0.000029                     slice_genstep 
                  upload genstep slice 1792498620.965950 1792498620.968551              2601                67                68          0.002601              upload genstep slice 
                        simulate slice 1792498620.968551 1792498655.068900          34100349                68                69         34.100349                    simulate slice 
                        download slice 1792498655.068900 1792498655.512901            444001                69                70          0.444001                    download slice 
                        concat slices  1792498655.512958 1792498655.512978                20                71                72          0.000020                    concat slices  
                          save arrays  1792498655.512991 1792498666.170623          10657632                73                78         10.657632                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792498487262349 2026-10-20T20:14:47.262
                                TOTAL: 59152452500220403 59152452679122180         178901777              1231              1281


hit array 1.3G
----------------

::

    A[blyth@localhost A000]$ du -h hit.npy 
    1.3G	hit.npy


Is "save arrays" really all array saving ?
----------------------------------------------

sreport::RANGES::

       A%0.3d_QSim__simulate_BRES:A%0.3d_QSim__simulate_TAIL         ## save arrays

::

     584     int64_t t_BRES  = SProf::Add("QSim__simulate_BRES", counts.c_str() );
     585     if(reset_) reset(eventID) ;   // reset calles SEvt::endOfEvent
     587     int64_t t_TAIL  = SProf::Add("QSim__simulate_TAIL");



     840 /**
     841 QSim::reset
     842 ------------
     843 
     844 When *QSim::simulate* is called with argument *reset:true* the
     845 *QSim::reset* method is called which invokes SEvt::endOfEvent in order
     846 to clean up the SEvt. Normally that would be done after saving
     847 any Opticks configured arrays.
     848 
     849 When *QSim::simulate* is called with argument *reset:false*
     850 the *QSim::reset* method must be called separately in order to avoid a memory leak.
     851 Using *reset:false* is typically done in order to keep arrays alive longer
     852 to enable copying from the gathered arrays into non-Opticks collections.
     853 
     854 **/
     855 void QSim::reset(int eventID)
     856 {
     857     SProf::Add("QSim__reset_HEAD");
     858     qev->clear();
     859     sev->endOfEvent(eventID);
     860     LOG_IF(info, SEvt::LIFECYCLE) << "] eventID " << eventID ;
     861     SProf::Add("QSim__reset_TAIL");
     862 }


     265 /**
     266 QEvt::clear
     267 --------------
     268 
     269 This is called from QSim::reset
     270 The former omission of gs deletion was reported by Ilker Parmaksiz.
     271 
     272 **/
     273 
     274 void QEvt::clear()
     275 {
     276     delete gs ;
     277     gs = nullptr ;
     278 }



