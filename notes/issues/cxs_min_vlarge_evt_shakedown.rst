cxs_min_vlarge_evt_shakedown
================================


::

    TEST=large_evt ~/o/cxs_min.sh 
    TEST=vlarge_evt ~/o/cxs_min.sh 
    TEST=vvlarge_evt ~/o/cxs_min.sh 

::

    334 elif [ "$TEST" == "vlarge_evt" ]; then
    335 
    336    opticks_num_photon=M500
    337    opticks_num_genstep=20
    338    opticks_num_event=1
    339    opticks_running_mode=SRM_TORCH
    340    #opticks_max_photon=M200        ## G1 default so no need to set  
    341    #opticks_max_slot=0              ## zero -> SEventConfig::SetDevice determines MaxSlot based on VRAM   
    342 

    343 elif [ "$TEST" == "vvlarge_evt" ]; then
    344 
    345    opticks_num_photon=G1
    346    opticks_num_genstep=40
    347    opticks_num_event=1
    348    opticks_running_mode=SRM_TORCH


Comment the max Philox defaults cover it::

      64 #if defined(RNG_XORWOW)
      65 const char* SEventConfig::_MaxCurandDefault = "M3" ; 
      66 const char* SEventConfig::_MaxSlotDefault = "M3" ;  
      67 const char* SEventConfig::_MaxGenstepDefault = "M3" ; 
      68 const char* SEventConfig::_MaxPhotonDefault = "M3" ;  
      69 const char* SEventConfig::_MaxSimtraceDefault = "M3" ; 
      70 
      71 #elif defined(RNG_PHILOX) || defined(RNG_PHILITEOX)
      72 const char* SEventConfig::_MaxCurandDefault = "G1" ; // nominal 1-billion states, as Philox has no need for curandState loading  
      73 const char* SEventConfig::_MaxSlotDefault = "0" ;     // see SEventConfig::SetDevice : set according to VRAM  
      74 const char* SEventConfig::_MaxGenstepDefault = "M10" ;  // adhoc  
      75 const char* SEventConfig::_MaxPhotonDefault = "G1" ; 
      76 const char* SEventConfig::_MaxSimtraceDefault = "G1" ;
      77 #endif



Ada managed 1 billion photons in 4 launches of 250M taking 2 min clocktime, kernel time less than 100s::

    2024-12-16 14:03:30.575 INFO  [56770] [QSim::simulate@385] sslice {    0,   10,      0,250000000}
    2024-12-16 14:03:56.724 INFO  [56770] [QSim::simulate@385] sslice {   10,   20,250000000,250000000}
    2024-12-16 14:04:23.235 INFO  [56770] [QSim::simulate@385] sslice {   20,   30,500000000,250000000}
    2024-12-16 14:04:49.998 INFO  [56770] [QSim::simulate@385] sslice {   30,   40,750000000,250000000}
    2024-12-16 14:05:29.785 INFO  [56770] [QSim::simulate@423]  eventID 0 tot_dt   94.502935 ph  250000000 ph/M        250 ht  215633111 ht/M        215 reset_ YES
    2024-12-16 14:05:29.785 INFO  [56770] [SEvt::save@3993] /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt/A000 [genstep,hit]



    2024-12-16 16:23:20.381 INFO  [62044] [QSim::simulate@391] sslice {    0,   10,      0,250000000}
    2024-12-16 16:23:46.508 INFO  [62044] [QSim::simulate@391] sslice {   10,   20,250000000,250000000}
    2024-12-16 16:24:13.038 INFO  [62044] [QSim::simulate@391] sslice {   20,   30,500000000,250000000}
    2024-12-16 16:24:39.798 INFO  [62044] [QSim::simulate@391] sslice {   30,   40,750000000,250000000}
    2024-12-16 16:25:18.928 INFO  [62044] [QSim::simulate@426]  eventID 0 tot_dt   94.236706 tot_ph 1000000000 tot_ph/M       1000 tot_ht  215633111 tot_ht/M        215 last_launch_num_ph  250000000 last_launch_num_ph/M        250 tot_ht/tot_ph          0 reset_ YES
    2024-12-16 16:25:18.929 INFO  [62044] [SEvt::save@3994] /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt/A000 [genstep,hit]
    2024-12-16 16:27:09.113 INFO  [62044] [QSim::simulate@443] 
    SEvt__MINTIME
     (LEND - LBEG)  106256689 (LEND - LBEG)/M        106 (multilaunch loop begin to end) 
     (PCAT - LEND)   12290570 (PCAT - LEND)/M         12 (topfold concat and clear subfold) 
     (TAIL - HEAD)  228732486 (TAIL - HEAD)/M        228 (head to tail of QSim::simulate method) 
     tot_idt   94236771 tot_idt/M           94       (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt  94.236706  int64_t(tot_dt*M)   94236705 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 





With 13G of hits::

    A[blyth@localhost CSGOptiX]$ TEST=vvlarge_evt ~/o/cxs_min.sh du
    13G /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt/A000/hit.npy




DONE : updated NP/NPFold/sreport metadata handling for multi-launch 
---------------------------------------------------------------------------




Ada /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt
--------------------------------------------------------------------------------------


::

    A[blyth@localhost ALL1_Debug_Philox_vvlarge_evt]$ JOB=II DEV=1 ~/opticks/sysrap/tests/sreport.sh
                            0 : /home/blyth/opticks/sysrap/tests/sreport.sh 
                  BASH_SOURCE : /home/blyth/opticks/sysrap/tests/sreport.sh 
                          arg : build_run_info_noa 
                       defarg : build_run_info_noa 
                          DEV : 1 
                          bin : /data1/blyth/tmp/sreport/sreport 
                       script : /home/blyth/opticks/sysrap/tests/sreport.py 
                         SDIR : /home/blyth/opticks/sysrap/tests 
                          JOB : II 
                          LAB : Undefined 
                          DIR : /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt 
                 SREPORT_FOLD : /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt_sreport 
                         MODE : 2 
                         name : sreport 
                         STEM : II__ 
                         PLOT :  
                         PICK :  
    [sreport.main  argv0 /data1/blyth/tmp/sreport/sreport dirp /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt is_executable_sibling_path NO 
    [sreport.main : CREATING REPORT 
    [sreport.main : creator 
    [sreport_Creator::sreport_Creator fold_valid YES run YES
    [sreport_Creator::init
    -sreport_Creator::init.1:runprof   :(2, 3, )
    -sreport_Creator::init.2.run       :(1, )
    -sreport_Creator::init.3.ranges2   :(14, 5, )
    -sreport_Creator::init.4 fold_valid Y
    -sreport_Creator::init.4.substamp   :[ subfold 1 ff 1 kk 0 aa 0]
    -sreport_Creator::init.5.subprofile :[ subfold 1 ff 1 kk 0 aa 0]
    -sreport_Creator::init.6.WITH_SUBMETA
    -sreport_Creator::init.7.submeta :[ subfold 0 ff 0 kk 2 aa 2]
    -sreport_Creator::init.8.submeta_NumPhotonCollected :[ subfold 0 ff 0 kk 2 aa 2]
    -sreport_Creator::init.9.subcount :[ subfold 0 ff 0 kk 2 aa 2]
    ]sreport_Creator::init
    ]sreport_Creator::sreport_Creator
    ]sreport.main : creator 
    [sreport.main : creator.desc 
    [sreport_Creator.desc
    [sreport_Creator.desc_fold
    fold = NPFold::LoadNoData("/data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt")
    fold YES
    fold_valid YES
    ]sreport_Creator.desc_fold
    ]sreport_Creator.desc
    ]sreport.main : creator.desc 
    [sreport.main : report.desc 
    [sreport.desc
    [sreport.desc_run (run is dummy small array used as somewhere to hang metadata) (1, )
    [sreport.desc_run.descMetaKVS 
    [NP::descMetaKVS
    [NP::DescMetaKVS only_with_stamp : NO 
    [NP::DescMetaKVS_kvs  keys.size 69 vals.size 69 tt.size 69 num_keys 69
                  OPTICKS_MAX_SLOT :                                                                                                      
                OPTICKS_MAX_PHOTON :                                                                                                      
              OPTICKS_OPTIX_PREFIX : /cvmfs/opticks.ihep.ac.cn/external/OptiX_800                                                                  
                OPTICKS_MAX_BOUNCE :                                  31                                                                  
              OPTICKS_RUNNING_MODE :                           SRM_TORCH                                                                  
                      OPTICKS_HOME :                 /home/blyth/opticks                                                                  
             OPTICKS_GEANT4_PREFIX : /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno                                                                  
                OPTICKS_EVENT_MODE :                                 Hit                                                                  
                OPTICKS_NUM_PHOTON :                                  G1                                                                  
                    OPTICKS_PREFIX :    /data1/blyth/local/opticks_Debug                                                                  
                            source :                   CSGOptiX__InitEvt                                                                  
                OPTICKS_MAX_CURAND :                                                                                                      
        OPTICKS_COMPUTE_CAPABILITY :                                  89                                                                  
               OPTICKS_START_INDEX :                                   0                                                                  
            OPTICKS_DOWNLOAD_CACHE : /cvmfs/opticks.ihep.ac.cn/opticks_download_cache                                                                  
                 OPTICKS_STTF_PATH : /data1/blyth/local/opticks_Debug/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf                                                                  
                           GPUMeta :    0:NVIDIA_RTX_5000_Ada_Generation                                                                  
                    QSim__Switches : CONFIG_Debug,NOT-CONFIG_RelWithDebInfo,NOT-CONFIG_Release,NOT-CONFIG_MinSizeRel,NOT-PRODUCTION,WITH_CHILD,WITH_CUSTOM4,PLOG_LOCAL,DEBUG_PIDX,DEBUG_TAG,NOT-RNG_XORWOW,RNG_PHILOX,NOT-RNG_PHILITEOX,                                                                  
                         C4Version :                                 TBD                                                                  
                 OPTICKS_BUILDTYPE :                               Debug                                                                  
                OPTICKS_EVENT_NAME :            Debug_Philox_vvlarge_evt                                                                  
                           creator :                      CSGOptiXSMTest                                                                  
                             uname : Linux localhost.localdomain 5.14.0-427.16.1.el9_4.x86_64 #1 SMP PREEMPT_DYNAMIC Thu May 9 18:15:59 EDT 2024 x86_64 x86_64 x86_64 GNU/Linux                                                                  
              CUDA_VISIBLE_DEVICES :                                   0                                                                  
                              HOME :                         /home/blyth                                                                  
                              USER :                               blyth                                                                  
                            SCRIPT :                          cxs_min.sh                                                                  
                               PWD : /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt                                                                  
                              TEST :                         vvlarge_evt                                                                  
                           VERSION :                                   1                                                                  
                              GEOM :                         J_2024aug27                                                                  
                  ${GEOM}_GEOMList :                J_2024aug27_GEOMList                                                                  
          OPTICKS_INTEGRATION_MODE :                                   1                                                                  
               OPTICKS_NUM_GENSTEP :                                  40                                                                  
              OPTICKS_EVENT_RELDIR : ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-none}                                                                  
               OPTICKS_CUDA_PREFIX :                /usr/local/cuda-12.4                                                                  
                 OPTICKS_NUM_EVENT :                                   1                                                                  

                                 k :                                   v                 t:microsecond  dt0:(t-t0)  dt1:(t-t1)  dt:(t-tpr)
               SEvt__Init_RUN_META :         1734446740501748,28368,7616    2024-12-17T22:45:40.501748           0                        
       CSGOptiX__SimulateMain_HEAD :        1734446740502375,28368,10304    2024-12-17T22:45:40.502375         627           0         627
             CSGFoundry__Load_HEAD :        1734446740502403,28368,10304    2024-12-17T22:45:40.502403         655          28          28
             CSGFoundry__Load_TAIL :     1734446744737592,5468996,753588    2024-12-17T22:45:44.737592   4,235,844   4,235,217   4,235,189
             CSGOptiX__Create_HEAD :     1734446744737613,5468996,753588    2024-12-17T22:45:44.737613   4,235,865   4,235,238          21
             CSGOptiX__Create_TAIL :    1734446745004423,8301148,1039300    2024-12-17T22:45:45.004423   4,502,675   4,502,048     266,810
          A000_QSim__simulate_HEAD :    1734446745004464,8301148,1039300    2024-12-17T22:45:45.004464   4,502,716   4,502,089          41
                  SEvt__BeginOfRun :    1734446745004480,8301148,1039300    2024-12-17T22:45:45.004480   4,502,732   4,502,105          16
     SEvt__beginOfEvent_FIRST_EGPU :    1734446745004502,8301148,1039300    2024-12-17T22:45:45.004502   4,502,754   4,502,127          22
               SEvt__setIndex_A000 :    1734446745004525,8301148,1039300    2024-12-17T22:45:45.004525   4,502,777   4,502,150          23
          A000_QSim__simulate_LBEG :    1734446745004715,8301148,1039300    2024-12-17T22:45:45.004715   4,502,967   4,502,340         190
          A000_QSim__simulate_PREL :   1734446745048766,29633116,1039748    2024-12-17T22:45:45.048766   4,547,018   4,546,391      44,051
          A000_QSim__simulate_POST :   1734446768186689,29633116,1044676    2024-12-17T22:46:08.186689  27,684,941  27,684,314  23,137,923
          A000_QSim__simulate_DOWN :   1734446772162556,33002664,4415088    2024-12-17T22:46:12.162556  31,660,808  31,660,181   3,975,867
          A000_QSim__simulate_PREL :   1734446772175168,33002664,4415088    2024-12-17T22:46:12.175168  31,673,420  31,672,793      12,612
          A000_QSim__simulate_POST :   1734446795624395,33002664,4415088    2024-12-17T22:46:35.624395  55,122,647  55,122,020  23,449,227
          A000_QSim__simulate_DOWN :   1734446799548499,36371712,7784136    2024-12-17T22:46:39.548499  59,046,751  59,046,124   3,924,104
          A000_QSim__simulate_PREL :   1734446799561125,36371712,7784136    2024-12-17T22:46:39.561125  59,059,377  59,058,750      12,626
          A000_QSim__simulate_POST :   1734446823297567,36371712,7784136    2024-12-17T22:47:03.297567  82,795,819  82,795,192  23,736,442
          A000_QSim__simulate_DOWN :  1734446827405882,39741632,11154056    2024-12-17T22:47:07.405882  86,904,134  86,903,507   4,108,315
          A000_QSim__simulate_PREL :  1734446827418540,39741632,11154056    2024-12-17T22:47:07.418540  86,916,792  86,916,165      12,658
          A000_QSim__simulate_POST :  1734446851269460,39741632,11154056    2024-12-17T22:47:31.269460 110,767,712 110,767,085  23,850,920
          A000_QSim__simulate_DOWN :  1734446855388735,43110196,14522504    2024-12-17T22:47:35.388735 114,886,987 114,886,360   4,119,275
          A000_QSim__simulate_LEND :  1734446855388773,43110196,14522504    2024-12-17T22:47:35.388773 114,887,025 114,886,398          38
          A000_QSim__simulate_PCAT :  1734446871288931,43110188,14522548    2024-12-17T22:47:51.288931 130,787,183 130,786,556  15,900,158
          A000_QSim__simulate_BRES :  1734446871289062,43110188,14522548    2024-12-17T22:47:51.289062 130,787,314 130,786,687         131
             A000_QSim__reset_HEAD :  1734446871289070,43110188,14522548    2024-12-17T22:47:51.289070 130,787,322 130,786,695           8
               SEvt__endIndex_A000 :  1734446871289088,43110188,14522548    2024-12-17T22:47:51.289088 130,787,340 130,786,713          18
                    SEvt__EndOfRun :   1734446988838729,29633116,1045476    2024-12-17T22:49:48.838729 248,336,981 248,336,354 117,549,641
             A000_QSim__reset_TAIL :   1734446988840448,29633116,1045476    2024-12-17T22:49:48.840448 248,338,700 248,338,073       1,719
          A000_QSim__simulate_TAIL :   1734446988840461,29633116,1045476    2024-12-17T22:49:48.840461 248,338,713 248,338,086          13
       CSGOptiX__SimulateMain_TAIL :   1734446988840540,29633116,1045476    2024-12-17T22:49:48.840540 248,338,792 248,338,165          79
    ]NP::DescMetaKVS_kvs
    [NP::DescMetaKVS_juncture
    num_juncture 4
    juncture [SEvt__Init_RUN_META,SEvt__BeginOfRun,SEvt__EndOfRun,SEvt__Init_RUN_META] time ranges between junctures
                                 k :          dtp                        :          dt0 : timestamp
               SEvt__Init_RUN_META :           -1                        :           -1 : 2024-12-17T22:45:40.501748 JUNCTURE
                  SEvt__BeginOfRun :    4,502,732                        :           -1 : 2024-12-17T22:45:45.004480 JUNCTURE
                    SEvt__EndOfRun :  243,834,249                        :           -1 : 2024-12-17T22:49:48.838729 JUNCTURE
               SEvt__Init_RUN_META : -248,336,981                        :           -1 : 2024-12-17T22:45:40.501748 JUNCTURE
    ]NP::DescMetaKVS_juncture
    [NP::DescMetaKVS_ranges2
    [ranges

            SEvt__Init_RUN_META:CSGFoundry__Load_HEAD                     ## init
            CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL                   ## load_geom
            CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL                   ## upload_geom
            A%0.3d_QSim__simulate_HEAD:A%0.3d_QSim__simulate_LBEG         ## slice_genstep
            A%0.3d_QSim__simulate_PRUP:A%0.3d_QSim__simulate_PREL         ## upload genstep slice
            A%0.3d_QSim__simulate_PREL:A%0.3d_QSim__simulate_POST         ## simulate slice
            A%0.3d_QSim__simulate_POST:A%0.3d_QSim__simulate_DOWN         ## download slice
            A%0.3d_QSim__simulate_LEND:A%0.3d_QSim__simulate_PCAT         ## concat slices
            A%0.3d_QSim__simulate_BRES:A%0.3d_QSim__simulate_TAIL         ## save arrays 
           ]ranges
    [NP::MakeMetaKVS_ranges2 num_keys:69
    [NP::Resolve_ranges
     num_keys :69 num_ranges :9
    ]NP::Resolve_ranges
     num_keys :69 num_ranges :9 num_specs :8
    [NP::MakeMetaKVS_ranges2_table num_specs 8
    .NP::MakeMetaKVS_ranges2_table kpp.size 14
                SEvt__Init_RUN_META ==>           CSGFoundry__Load_HEAD                   655                               ## init
              CSGFoundry__Load_HEAD ==>           CSGFoundry__Load_TAIL             4,235,189                               ## load_geom
              CSGOptiX__Create_HEAD ==>           CSGOptiX__Create_TAIL               266,810                               ## upload_geom
           A000_QSim__simulate_HEAD ==>        A000_QSim__simulate_LBEG                   251                               ## slice_genstep
           A000_QSim__simulate_PREL ==>        A000_QSim__simulate_POST            23,137,923                               ## simulate slice
           A000_QSim__simulate_POST ==>        A000_QSim__simulate_DOWN             3,975,867                               ## download slice
           A000_QSim__simulate_PREL ==>        A000_QSim__simulate_POST            23,449,227 REP             46,587,150    ## simulate slice
           A000_QSim__simulate_POST ==>        A000_QSim__simulate_DOWN             3,924,104 REP              7,899,971    ## download slice
           A000_QSim__simulate_PREL ==>        A000_QSim__simulate_POST            23,736,442 REP             70,323,592    ## simulate slice
           A000_QSim__simulate_POST ==>        A000_QSim__simulate_DOWN             4,108,315 REP             12,008,286    ## download slice
           A000_QSim__simulate_PREL ==>        A000_QSim__simulate_POST            23,850,920 REP             94,174,512    ## simulate slice
           A000_QSim__simulate_POST ==>        A000_QSim__simulate_DOWN             4,119,275 REP             16,127,561    ## download slice
           A000_QSim__simulate_LEND ==>        A000_QSim__simulate_PCAT            15,900,158                               ## concat slices
           A000_QSim__simulate_BRES ==>        A000_QSim__simulate_TAIL           117,551,399                               ## save arrays
                                                                 TOTAL:           248,256,535
    ]NP::MakeMetaKVS_ranges2_table num_keys:69
    ]NP::MakeMetaKVS_ranges2 num_specs:8 rr (14, 5, )
    ]NP::DescMetaKVS_ranges2 a (14, 5, )
    ]NP::DescMetaKVS
    ]NP::descMetaKVS

    ]sreport.desc_run.descMetaKVS 
    ]sreport.desc_run
    [sreport.desc_runprof
    (2, 3, )
    .sreport.desc_runprof.descTable 
    [NP::descTable_ (2, 3, )
                                                  st[us]            vm[kb]            rs[kb]
                   SEvt__setIndex_A000          0.000000           8301148           1039300
                   SEvt__endIndex_A000        126.284563          43110188          14522548
    num_timestamp 2 auto-offset from t0 1734446745004525
                                TOTAL:         126284563          51411336          15561848
    ]NP::descTable_ (2, 3, )

    ]sreport.desc_runprof
    [sreport.desc_ranges ranges : (14, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ (14, 5, )
                                                      ta                tb                ab                ia                ib
                                SIRMLH          0.000000          0.000655               655                 0                 7
                                CLHLTg          0.000655          4.235844           4235189                 7                 8
                              CXCHXCTg          4.235865          4.502675            266810                 9                10
                              AQsHQsLg          4.502716          4.502967               251                11                12
                               AQsPQsP          4.547018         27.684941          23137923                13                14
                               AQsPQsD         27.684941         31.660808           3975867                14                15
                               AQsPQsP         31.673420         55.122647          23449227                16                17
                               AQsPQsD         55.122647         59.046751           3924104                17                18
                               AQsPQsP         59.059377         82.795819          23736442                19                20
                               AQsPQsD         82.795819         86.904134           4108315                20                21
                               AQsPQsP         86.916792        110.767712          23850920                22                23
                               AQsPQsD        110.767712        114.886987           4119275                23                24
                               AQsLQsP        114.887025        130.787183          15900158                25                26
                               AQsBQsT        130.787314        248.338713         117551399                27                30
    num_timestamp 28 auto-offset from t0 1734446740501748
                                TOTAL:         712981301         961237836         248256535               223               245

                                SIRMLH : SEvt__Init_RUN_META:CSGFoundry__Load_HEAD:init
                                CLHLTg : CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL:load_geom
                              CXCHXCTg : CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL:upload_geom
                              AQsHQsLg : A000_QSim__simulate_HEAD:A000_QSim__simulate_LBEG:slice_genstep
                               AQsPQsP : A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
                               AQsPQsD : A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
                               AQsPQsP : A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
                               AQsPQsD : A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
                               AQsPQsP : A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
                               AQsPQsD : A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
                               AQsPQsP : A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
                               AQsPQsD : A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
                               AQsLQsP : A000_QSim__simulate_LEND:A000_QSim__simulate_PCAT:concat slices
                               AQsBQsT : A000_QSim__simulate_BRES:A000_QSim__simulate_TAIL:save arrays
    ]NP::descTable_ (14, 5, )

    ]sreport.desc_ranges
    [sreport.desc_substamp
    [sreport.desc_substamp.compare_subarrays_report
    NPX::BOA A (1, 13, ) B -
    NPX::BOA ABORT A or B null 
    [NPFold::compare_subarray key delta_substamp asym a bsym b af YES bf NO  a YES b NO  a_subcount YES b_subcount NO  boa NO 
    -[NPFold::compare_subarray.a_subcount
    [NP::descTable_ (1, 2, )
                      genstep      hit
              //A000       40 215633111
    num_timestamp 0 auto-offset from t0 0
              TOTAL:       40 215633111
    ]NP::descTable_ (1, 2, )
    -]NPFold::compare_subarray.a_subcount
    -[NPFold::compare_subarray.b_subcount
    -
    -]NPFold::compare_subarray.b_subcount
    -[NPFold::compare_subarray.a
    [NP::descTable_ (1, 13, )
                        SbOE0    SbOE1    SeOE0     tBOE     tsG3     tsG4     tsG5     tsG6     tsG7     tsG8     tPrL     tPoL     tEOE
              //A000        0      172 126284565        6 82401462 82401463 82401474 82401491 82414021 82414027 82414042 106264947 126284573
    num_timestamp 0 auto-offset from t0 0
              TOTAL:        0      172 126284565        6 82401462 82401463 82401474 82401491 82414021 82414027 82414042 106264947 126284573

       SbOE0 : SEvt__beginOfEvent_0
       SbOE1 : SEvt__beginOfEvent_1
       SeOE0 : SEvt__endOfEvent_0
        tBOE : t_BeginOfEvent
        tsG3 : t_setGenstep_3
        tsG4 : t_setGenstep_4
        tsG5 : t_setGenstep_5
        tsG6 : t_setGenstep_6
        tsG7 : t_setGenstep_7
        tsG8 : t_setGenstep_8
        tPrL : t_PreLaunch
        tPoL : t_PostLaunch
        tEOE : t_EndOfEvent
    ]NP::descTable_ (1, 13, )
    -]NPFold::compare_subarray.a
    -[NPFold::compare_subarray.b
    -
    -]NPFold::compare_subarray.b
    -[NPFold::compare_subarray.boa 
    -
    -]NPFold::compare_subarray.boa 
    ]NPFold::compare_subarray
    ]sreport.desc_substamp.compare_subarrays_report
    ]sreport.desc_substamp
    [sreport.desc_submeta
    [NPFold::desc
    NPFold::subfold_summary("submeta","a://A","b://B" )
    NPFold::desc_subfold
     tot_items 2
     folds 1
     paths 1
      0 [/]  stamp:0
    NPFold::desc(0) 
    NPFold::desc( 0)
     subfold 0 ff 0 kk 2 aa 2
    [NP::DescMetaKVS only_with_stamp : NO 
    ]NP::DescMetaKVS

                      a.npy : (1, )
                      b.npy : (0, )

    ]NPFold::desc
    ]sreport.desc_submeta
    [sreport.desc_subcount
    [NPFold::desc
    NPFold::subfold_summary("subcount","a://A","b://B" )
    NPFold::desc_subfold
     tot_items 2
     folds 1
     paths 1
      0 [/]  stamp:0
    NPFold::desc(0) 
    NPFold::desc( 0)
     subfold 0 ff 0 kk 2 aa 2
    [NP::DescMetaKVS only_with_stamp : NO 
    ]NP::DescMetaKVS

                      a.npy : (1, 2, )
                      b.npy : (0, )

    ]NPFold::desc
    ]sreport.desc_subcount
    ]sreport.desc
    ]sreport.main : report.desc 
    NPFold::save("$SREPORT_FOLD")
     resolved to  [/data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt_sreport]
    ]sreport.main : CREATED REPORT 
    ]sreport.main
    /home/blyth/opticks/sysrap/tests/sreport.sh : noa : no analysis exit
    A[blyth@localhost ALL1_Debug_Philox_vvlarge_evt]$ 

