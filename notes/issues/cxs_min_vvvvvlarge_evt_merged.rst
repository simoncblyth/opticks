cxs_min_vvvvvlarge_evt_merged
==============================

Overview
---------

* FIXED : photonlite/hitlite running has a whopper memory leak, as shown by nvtop and death on 8th-of-32 slice of 8.25 billion test

  * the latch did not handle photonlite causing repeated calls to : QEvt::device_alloc_photon

* BUT : smaller leak remains, see step of about 900MB about every 70 sec, giving abort on 25th-of-32 slice of 8.25 billion test

   

* suspect photon being alloc and not used 
* suspect photonlite being alloc and not cleaned




Smaller leak : with steps of 0.9 GB  in nvtop
-----------------------------------------------

Leak size points to hitlite + hitlitemerged::

    In [3]: 260*1e6*4*4/1e9 # size of photonlite max_slot in each launch
    Out[3]: 4.16

    In [4]: 260*1e6*4*4/1e9*0.1999 # size of hitlite in each launch
    Out[4]: 0.831584

    In [5]: 260*1e6*4*4/1e9*0.1999*0.1 # size of hitlitemerged in each launch
    Out[5]: 0.08315840000000001

    In [6]: 260*1e6*4*4/1e9*0.1999*1.1 # size of hitlite + hitlitemerged in each launch [GB]
    Out[6]: 0.9147424000000001


::

    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51436148 merged 5512106 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:40:16.305 INFO  [3578084] [QSim::simulate@479]   19 : sslice {     304,     320,4900092096, 257899584}257.899584
    2025-11-19 21:40:55.295 INFO  [3578084] [QSim::simulate@505]  eventID 0 xxl YES i   19 dt   38.981315 slice   19 : sslice {     304,     320,4900092096, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51438127 merged 5512713 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:40:55.511 INFO  [3578084] [QSim::simulate@479]   20 : sslice {     320,     336,5157991680, 257899584}257.899584
    2025-11-19 21:41:34.731 INFO  [3578084] [QSim::simulate@505]  eventID 0 xxl YES i   20 dt   39.212780 slice   20 : sslice {     320,     336,5157991680, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51436702 merged 5513186 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:41:34.965 INFO  [3578084] [QSim::simulate@479]   21 : sslice {     336,     352,5415891264, 257899584}257.899584
    2025-11-19 21:42:13.960 INFO  [3578084] [QSim::simulate@505]  eventID 0 xxl YES i   21 dt   38.988601 slice   21 : sslice {     336,     352,5415891264, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51442011 merged 5512866 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:42:14.186 INFO  [3578084] [QSim::simulate@479]   22 : sslice {     352,     368,5673790848, 257899584}257.899584
    2025-11-19 21:42:53.284 INFO  [3578084] [QSim::simulate@505]  eventID 0 xxl YES i   22 dt   39.091316 slice   22 : sslice {     352,     368,5673790848, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51434476 merged 5512226 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:42:53.515 INFO  [3578084] [QSim::simulate@479]   23 : sslice {     368,     384,5931690432, 257899584}257.899584
    2025-11-19 21:43:32.407 INFO  [3578084] [QSim::simulate@505]  eventID 0 xxl YES i   23 dt   38.886316 slice   23 : sslice {     368,     384,5931690432, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51450523 merged 5515626 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:43:32.640 INFO  [3578084] [QSim::simulate@479]   24 : sslice {     384,     400,6189590016, 257899584}257.899584
    2025-11-19 21:44:11.609 INFO  [3578084] [QSim::simulate@505]  eventID 0 xxl YES i   24 dt   38.961020 slice   24 : sslice {     384,     400,6189590016, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51437107 merged 5512405 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:44:11.847 INFO  [3578084] [QSim::simulate@479]   25 : sslice {     400,     416,6447489600, 257899584}257.899584
    2025-11-19 21:44:50.813 INFO  [3578084] [QSim::simulate@505]  eventID 0 xxl YES i   25 dt   38.960020 slice   25 : sslice {     400,     416,6447489600, 257899584}257.899584

    Thread 1 "CSGOptiXSMTest" received signal SIGABRT, Aborted.
    0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6



Where is hit/hitlite/hitlitemerged alloc and cleanup done ?
------------------------------------------------------------

* hit cleanup done in  QEvt::gatherHit_() immediately after download to host



After fixed leaks, succeed with 8.25 billion event : using hitlitemerged
--------------------------------------------------------------------------

::

    (ok) A[blyth@localhost CSGOptiX]$ cxs_min.sh
    BASH_SOURCE                    : /data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    SDIR                           : /data1/blyth/local/opticks_Debug/bin 
    defarg                         : run_report_info 
    arg                            : run_report_info 
    allarg                         : info_env_fold_run_dbg_meta_report_grab_grep_gevt_du_pdb1_pdb0_AB_ana_pvcap_pvpub_mpcap_mppub 
    bin                            : CSGOptiXSMTest 
    script                         : /data1/blyth/local/opticks_Debug/bin/cxs_min.py 
    script_AB                      : /data1/blyth/local/opticks_Debug/bin/cxs_min_AB.py 
    script_lite                    : /data1/blyth/local/opticks_Debug/bin/cxs_min_lite.py 
    GEOM                           : J25_4_0_opticks_Debug 
    TMP                            : /data1/blyth/tmp 
    EVT                            : A000 
    BASE                           : /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug 
    BINBASE                        : /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest 
    SCRIPT                         : cxs_min.sh 
    J25_4_0_opticks_Debug_CFBaseFromGEOM : /home/blyth/junosw/InstallArea/.opticks/GEOM/J25_4_0_opticks_Debug 
    J25_4_0_opticks_Debug_GDMLPathFromGEOM :  
    version                        : 1 
    VERSION                        : 1 
    test                           : vvvvvvlarge_evt_merge 
    TEST                           : vvvvvvlarge_evt_merge 
    opticks_event_reldir           : ALL1_Debug_Philox_vvvvvvlarge_evt_merge 
    OPTICKS_EVENT_RELDIR           : ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-none} 
    alt_TEST                       :  
    alt_opticks_event_reldir       : ALL1_ 
    LOGDIR                         : /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvvlarge_evt_merge 
    LOGDIR0                        :  
    LOGOVER                        :  
    AFOLD                          : /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvvlarge_evt_merge/A000 
    BFOLD                          :  
    BFOLD_NOTE                     : defining BFOLD makes python script do SAB comparison 
    STEM                           : ALL1_Debug_Philox_vvvvvvlarge_evt_merge_ 
    LOGFILE                        : CSGOptiXSMTest.log 
    opticks_event_mode             : Hit 
    opticks_num_event              : 1 
    opticks_num_genstep            : 512 
    opticks_num_photon             : 8252787186 
    opticks_running_mode           : SRM_TORCH 
    opticks_max_slot               :  
    version                        : 1 
    VERSION                        : 1 
    opticks_event_mode             : Hit 
    OPTICKS_EVENT_MODE             : Hit 
    OPTICKS_NUM_PHOTON             : 8252787186 
    OPTICKS_NUM_GENSTEP            : 512 
    OPTICKS_MAX_PHOTON             :  
    OPTICKS_NUM_EVENT              : 1 
    OPTICKS_RUNNING_MODE           : SRM_TORCH 
    OPTICKS_MODE_LITE              : 1 
    OPTICKS_MODE_LITE_NOTE         : photonlite/hitlite variants will be saved instead of photon/hit (if they are configured) 
    OPTICKS_MODE_MERGE             : 1 
    OPTICKS_MODE_MERGE_NOTE        : ModeLite is configured then attempt GPU side hitlite merge yielding hitlitemerged array 
    opticks_merge_window           : 1 
    OPTICKS_MERGE_WINDOW           : 1 
    OPTICKS_MAX_CURAND             :  
    OPTICKS_MAX_SLOT               :  
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
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2025-11-19 22:29:23.876  876522795 : [/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    SLOG::EnvLevel adjusting loglevel by envvar   key QSim level INFO fallback DEBUG upper_level INFO
    2025-11-19 22:29:25.280 INFO  [3599279] [SEventConfig::SetDevice@1785] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262326496
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 0

    2025-11-19 22:29:25.281 INFO  [3599279] [SEventConfig::SetDevice@1797]  Configured_MaxSlot/M 0 Final_MaxSlot/M 262 HeuristicMaxSlot_Rounded/M 262 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2025-11-19 22:29:25.424 INFO  [3599279] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-19 22:29:25.432 INFO  [3599279] [QSim::UploadComponents@118] [ ssim 0xddd990
    2025-11-19 22:29:25.432 INFO  [3599279] [QSim::UploadComponents@121] [ new QBase
    2025-11-19 22:29:25.533 INFO  [3599279] [QSim::UploadComponents@123] ] new QBase : latency here of about 0.3s from first device access, if latency of >1s need to start nvidia-persistenced 
    2025-11-19 22:29:25.533 INFO  [3599279] [QSim::UploadComponents@124] QBase::desc base 0x17080dc0 d_base 0x7ffb8ec00000 base.desc qbase::desc pidx 1099511627775
    2025-11-19 22:29:25.533 INFO  [3599279] [QSim::UploadComponents@128] [ new QRng skipahead_event_offset : 100000 OPTICKS_EVENT_SKIPAHEAD
    2025-11-19 22:29:25.533 INFO  [3599279] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000000 SEventConfig::MaxCurand 1000000000000
    2025-11-19 22:29:25.533 INFO  [3599279] [QRng::init@104] [QRng__init_VERBOSE] YES
    QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 1000000000000
                         rngmax/M 1000000
                               qr 0x18a25400
        qr.skipahead_event_offset 100000
                             d_qr 0x7ffb8ec00200

    2025-11-19 22:29:25.533 INFO  [3599279] [QSim::UploadComponents@130] ] new QRng QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 1000000000000
                         rngmax/M 1000000
                               qr 0x18a25400
        qr.skipahead_event_offset 100000
                             d_qr 0x7ffb8ec00200

    2025-11-19 22:29:25.533 INFO  [3599279] [QSim::UploadComponents@132] QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 1000000000000
                         rngmax/M 1000000
                               qr 0x18a25400
        qr.skipahead_event_offset 100000
                             d_qr 0x7ffb8ec00200

    2025-11-19 22:29:25.536 INFO  [3599279] [QSim::UploadComponents@145] QOptical optical NP  dtype <i4(324, 4, 4, ) size 5184 uifc i ebyte 4 shape.size 3 data.size 20736 meta.size 0 names.size 0
    2025-11-19 22:29:25.579 INFO  [3599279] [QSim::UploadComponents@148] QBnd src NP  dtype <f4(324, 4, 2, 761, 4, ) size 7890048 uifc f ebyte 4 shape.size 5 data.size 31560192 meta.size 61 names.size 324 tex QTex width 761 height 2592 texObj 1 meta 0x18a3adb0 d_meta 0x7ffb8ec05600 tex 0x18a3aed0
    2025-11-19 22:29:25.579 INFO  [3599279] [QSim::UploadComponents@152] QDebug::desc  dbg 0x18a328b0 d_dbg 0x7ffb8ec05a00
     QState::Desc QState::Desc
    material1 ( 1.000,1000.000,1000.000, 0.000) 
    material2 ( 1.500,1000.000,1000.000, 0.000) 
    m1group2  (300.000, 0.000, 0.000, 0.000) 
    surface   ( 0.000, 0.000, 0.000, 0.000) 
    optical   (     0,     0,     0,     0) 

     dbg.p.desc  pos ( 0.000, 0.000, 0.000)  t     0.000  mom ( 1.000, 0.000, 0.000)  iindex 1065353216  pol ( 0.000, 1.000, 0.000)  wl  500.000   bn 0 fl 0 id 0 or 1 ix 0 fm 0 ab    ii 1065353216
    2025-11-19 22:29:25.581 INFO  [3599279] [QSim::UploadComponents@165]   propcom null, snam::PROPCOM propcom.npy
    2025-11-19 22:29:25.582 INFO  [3599279] [QSim::UploadComponents@178] QScint dsrc NP  dtype <f8(3, 4096, 1, ) size 12288 uifc f ebyte 8 shape.size 3 data.size 98304 meta.size 97 names.size 1 src NP  dtype <f4(3, 4096, 1, ) size 12288 uifc f ebyte 4 shape.size 3 data.size 49152 meta.size 97 names.size 1 tex QTex width 4096 height 3 texObj 2 meta 0x18a31340 d_meta 0x7ffb8ec05c00 tex 0x18a3e0b0
    2025-11-19 22:29:25.582 INFO  [3599279] [QSim::UploadComponents@187] QCerenkov fold - icdf_ - icdf - tex 0
    SPMT::init_pmtNum pmtNum.sstr (6, )
    [SPMT::init_pmtNum
    [SPMT_Num.desc
    m_nums_cd_lpmt     :  17612
    m_nums_cd_spmt     :  25600
    m_nums_wp_lpmt     :   2400
    m_nums_wp_atm_lpmt :    348
    m_nums_wp_wal_pmt  :      5
    m_nums_wp_atm_mpmt :    600
    ALL                :  46565
    SUM                :  46565
    SUM==ALL           :    YES
    ]SPMT_Num.desc
    ]SPMT::init_pmtNum
    2025-11-19 22:29:25.596 INFO  [3599279] [QSim::UploadComponents@217] QPMT<float> WITH_CUSTOM4  INSTANCE:YES QPMT::desc
                           rindex (24, 15, 2, )
                          qeshape (3, 44, 2, )
                          cetheta (3, 10, 2, )
                          cecosth (3, 10, 2, )
                        thickness (3, 4, 1, )
                             lcqs (20965, 2, )
                        s_qeshape (1, 61, 2, )
                        s_qescale (25600, 1, )
                  pmt.rindex_prop 0x7ffb8ec06e00
                 pmt.qeshape_prop 0x7ffb8ec07600
                 pmt.cetheta_prop 0x7ffb8ec07a00
                 pmt.cecosth_prop 0x7ffb8ec07e00
                    pmt.thickness 0x7ffb8ec08400
                         pmt.lcqs 0x7ffb8ec08600
                            d_pmt 0x7ffb8ec4a600

     spmt_f YES qpmt YES
    2025-11-19 22:29:25.596 INFO  [3599279] [QSim::UploadComponents@229]  multifilm null, snam::MULTIFILM multifilm.npy
    2025-11-19 22:29:25.596 INFO  [3599279] [QSim::UploadComponents@236] ] ssim 0xddd990
    2025-11-19 22:29:25.596 INFO  [3599279] [QSim::QSim@280] QSim::desc
     this 0x18ff6a80 INSTANCE 0x0 QEvt.hh:qev 0x18ff6a20 qsim.h:sim 0x0
    2025-11-19 22:29:25.596 INFO  [3599279] [QSim::init@324]  MISSING_PMT NO  has_PMT YES QSim::pmt YES QSim::pmt->d_pmt YES [QSim__REQUIRE_PMT] NO 
    2025-11-19 22:29:25.596 INFO  [3599279] [QSim::init@347] QSim::desc
     this 0x18ff6a80 INSTANCE 0x18ff6a80 QEvt.hh:qev 0x18ff6a20 qsim.h:sim 0x18ff46f0
    2025-11-19 22:29:25.596 INFO  [3599279] [QSim::init@348] 
    QSim::descComponents
     (QBase)base             YES
     (QEvt)qev           YES
     (SEvt)sev               YES
     (QRng)rng               YES
     (QScint)scint           YES
     (QCerenkov)cerenkov     YES
     (QBnd)bnd               YES
     (QOptical)optical       NO 
     (QDebug)debug_          YES
     (QProp)prop             YES
     (QPMT)pmt               YES
     (QMultiFilm)multifilm   NO 
     (qsim)sim               YES
     (qsim)d_sim             YES
     (qdebug)dbg             YES
     (qdebug)d_dbg           YES

    2025-11-19 22:29:25.876 INFO  [3599279] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-19 22:29:25.878 INFO  [3599279] [QSim::MaybeSaveIGS@729]  eventID 0 igs (512, 6, 4, ) igs_null NO  [QSim__SAVE_IGS_EVENTID] -1 [QSim__SAVE_IGS_PATH] $TMP/.opticks/igs.npy igs_path [/data1/blyth/tmp/.opticks/igs.npy] save_igs NO 
    2025-11-19 22:29:25.878 INFO  [3599279] [QSim::simulate@457]  eventID      0 igs (512, 6, 4, ) tot_ph_0 8252786688 tot_ph_0/M 8252 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 32 TotalPhoton 8252786688 TotalPhoton/M 8252.786688
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      16,         0, 257899584}257.899584
       1 : sslice {      16,      32, 257899584, 257899584}257.899584
       2 : sslice {      32,      48, 515799168, 257899584}257.899584
       3 : sslice {      48,      64, 773698752, 257899584}257.899584
       4 : sslice {      64,      80,1031598336, 257899584}257.899584
       5 : sslice {      80,      96,1289497920, 257899584}257.899584
       6 : sslice {      96,     112,1547397504, 257899584}257.899584
       7 : sslice {     112,     128,1805297088, 257899584}257.899584
       8 : sslice {     128,     144,2063196672, 257899584}257.899584
       9 : sslice {     144,     160,2321096256, 257899584}257.899584
      10 : sslice {     160,     176,2578995840, 257899584}257.899584
      11 : sslice {     176,     192,2836895424, 257899584}257.899584
      12 : sslice {     192,     208,3094795008, 257899584}257.899584
      13 : sslice {     208,     224,3352694592, 257899584}257.899584
      14 : sslice {     224,     240,3610594176, 257899584}257.899584
      15 : sslice {     240,     256,3868493760, 257899584}257.899584
      16 : sslice {     256,     272,4126393344, 257899584}257.899584
      17 : sslice {     272,     288,4384292928, 257899584}257.899584
      18 : sslice {     288,     304,4642192512, 257899584}257.899584
      19 : sslice {     304,     320,4900092096, 257899584}257.899584
      20 : sslice {     320,     336,5157991680, 257899584}257.899584
      21 : sslice {     336,     352,5415891264, 257899584}257.899584
      22 : sslice {     352,     368,5673790848, 257899584}257.899584
      23 : sslice {     368,     384,5931690432, 257899584}257.899584
      24 : sslice {     384,     400,6189590016, 257899584}257.899584
      25 : sslice {     400,     416,6447489600, 257899584}257.899584
      26 : sslice {     416,     432,6705389184, 257899584}257.899584
      27 : sslice {     432,     448,6963288768, 257899584}257.899584
      28 : sslice {     448,     464,7221188352, 257899584}257.899584
      29 : sslice {     464,     480,7479087936, 257899584}257.899584
      30 : sslice {     480,     496,7736987520, 257899584}257.899584
      31 : sslice {     496,     512,7994887104, 257899584}257.899584
                      start    stop     offset      count    count/M 
     num_slice 32
    2025-11-19 22:29:25.878 INFO  [3599279] [QSim::simulate@479]    0 : sslice {       0,      16,         0, 257899584}257.899584
    2025-11-19 22:30:03.406 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    0 dt   37.512372 slice    0 : sslice {       0,      16,         0, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51442248 merged 5511645 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:30:03.629 INFO  [3599279] [QSim::simulate@479]    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    2025-11-19 22:30:42.003 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    1 dt   38.367798 slice    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51430837 merged 5514117 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:30:42.222 INFO  [3599279] [QSim::simulate@479]    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    2025-11-19 22:31:21.021 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    2 dt   38.792436 slice    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51433012 merged 5515333 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:31:21.251 INFO  [3599279] [QSim::simulate@479]    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    2025-11-19 22:32:00.238 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    3 dt   38.979930 slice    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51449616 merged 5514339 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:32:00.463 INFO  [3599279] [QSim::simulate@479]    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    2025-11-19 22:32:39.462 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    4 dt   38.991948 slice    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51435671 merged 5512175 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:32:39.692 INFO  [3599279] [QSim::simulate@479]    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    2025-11-19 22:33:18.915 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    5 dt   39.216126 slice    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51441825 merged 5513278 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:33:19.143 INFO  [3599279] [QSim::simulate@479]    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    2025-11-19 22:33:58.567 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    6 dt   39.417163 slice    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51444895 merged 5513233 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:33:58.798 INFO  [3599279] [QSim::simulate@479]    7 : sslice {     112,     128,1805297088, 257899584}257.899584
    2025-11-19 22:34:38.136 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    7 dt   39.331325 slice    7 : sslice {     112,     128,1805297088, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51446156 merged 5513354 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:34:38.373 INFO  [3599279] [QSim::simulate@479]    8 : sslice {     128,     144,2063196672, 257899584}257.899584
    2025-11-19 22:35:17.470 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    8 dt   39.090278 slice    8 : sslice {     128,     144,2063196672, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51436479 merged 5513088 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:35:17.692 INFO  [3599279] [QSim::simulate@479]    9 : sslice {     144,     160,2321096256, 257899584}257.899584
    2025-11-19 22:35:56.826 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i    9 dt   39.127651 slice    9 : sslice {     144,     160,2321096256, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51444573 merged 5512930 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:35:57.047 INFO  [3599279] [QSim::simulate@479]   10 : sslice {     160,     176,2578995840, 257899584}257.899584
    2025-11-19 22:36:36.173 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   10 dt   39.118857 slice   10 : sslice {     160,     176,2578995840, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51444367 merged 5512454 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:36:36.401 INFO  [3599279] [QSim::simulate@479]   11 : sslice {     176,     192,2836895424, 257899584}257.899584
    2025-11-19 22:37:15.573 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   11 dt   39.165182 slice   11 : sslice {     176,     192,2836895424, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51447833 merged 5512496 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:37:15.806 INFO  [3599279] [QSim::simulate@479]   12 : sslice {     192,     208,3094795008, 257899584}257.899584
    2025-11-19 22:37:54.989 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   12 dt   39.176084 slice   12 : sslice {     192,     208,3094795008, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51438173 merged 5514237 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:37:55.221 INFO  [3599279] [QSim::simulate@479]   13 : sslice {     208,     224,3352694592, 257899584}257.899584
    2025-11-19 22:38:34.400 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   13 dt   39.171429 slice   13 : sslice {     208,     224,3352694592, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51439531 merged 5512109 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:38:34.633 INFO  [3599279] [QSim::simulate@479]   14 : sslice {     224,     240,3610594176, 257899584}257.899584
    2025-11-19 22:39:14.035 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   14 dt   39.395862 slice   14 : sslice {     224,     240,3610594176, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51444862 merged 5513475 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:39:14.272 INFO  [3599279] [QSim::simulate@479]   15 : sslice {     240,     256,3868493760, 257899584}257.899584
    2025-11-19 22:39:53.784 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   15 dt   39.505560 slice   15 : sslice {     240,     256,3868493760, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51435489 merged 5515275 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:39:54.008 INFO  [3599279] [QSim::simulate@479]   16 : sslice {     256,     272,4126393344, 257899584}257.899584
    2025-11-19 22:40:33.267 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   16 dt   39.251783 slice   16 : sslice {     256,     272,4126393344, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51448288 merged 5512037 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:40:33.506 INFO  [3599279] [QSim::simulate@479]   17 : sslice {     272,     288,4384292928, 257899584}257.899584
    2025-11-19 22:41:13.111 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   17 dt   39.598892 slice   17 : sslice {     272,     288,4384292928, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51443913 merged 5514566 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:41:13.339 INFO  [3599279] [QSim::simulate@479]   18 : sslice {     288,     304,4642192512, 257899584}257.899584
    2025-11-19 22:41:52.572 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   18 dt   39.226559 slice   18 : sslice {     288,     304,4642192512, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51436148 merged 5512106 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:41:52.794 INFO  [3599279] [QSim::simulate@479]   19 : sslice {     304,     320,4900092096, 257899584}257.899584
    2025-11-19 22:42:31.986 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   19 dt   39.185852 slice   19 : sslice {     304,     320,4900092096, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51438127 merged 5512713 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:42:32.206 INFO  [3599279] [QSim::simulate@479]   20 : sslice {     320,     336,5157991680, 257899584}257.899584
    2025-11-19 22:43:11.384 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   20 dt   39.171125 slice   20 : sslice {     320,     336,5157991680, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51436702 merged 5513186 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:43:11.619 INFO  [3599279] [QSim::simulate@479]   21 : sslice {     336,     352,5415891264, 257899584}257.899584
    2025-11-19 22:43:51.139 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   21 dt   39.512649 slice   21 : sslice {     336,     352,5415891264, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51442011 merged 5512866 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:43:51.409 INFO  [3599279] [QSim::simulate@479]   22 : sslice {     352,     368,5673790848, 257899584}257.899584
    2025-11-19 22:44:30.746 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   22 dt   39.329715 slice   22 : sslice {     352,     368,5673790848, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51434476 merged 5512226 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:44:30.973 INFO  [3599279] [QSim::simulate@479]   23 : sslice {     368,     384,5931690432, 257899584}257.899584
    2025-11-19 22:45:10.031 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   23 dt   39.050927 slice   23 : sslice {     368,     384,5931690432, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51450523 merged 5515626 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:45:10.267 INFO  [3599279] [QSim::simulate@479]   24 : sslice {     384,     400,6189590016, 257899584}257.899584
    2025-11-19 22:45:49.368 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   24 dt   39.094541 slice   24 : sslice {     384,     400,6189590016, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51437107 merged 5512405 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:45:49.592 INFO  [3599279] [QSim::simulate@479]   25 : sslice {     400,     416,6447489600, 257899584}257.899584
    2025-11-19 22:46:28.815 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   25 dt   39.215949 slice   25 : sslice {     400,     416,6447489600, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51434368 merged 5515092 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:46:29.044 INFO  [3599279] [QSim::simulate@479]   26 : sslice {     416,     432,6705389184, 257899584}257.899584
    2025-11-19 22:47:08.225 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   26 dt   39.174248 slice   26 : sslice {     416,     432,6705389184, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51447031 merged 5512717 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:47:08.451 INFO  [3599279] [QSim::simulate@479]   27 : sslice {     432,     448,6963288768, 257899584}257.899584
    2025-11-19 22:47:47.592 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   27 dt   39.134748 slice   27 : sslice {     432,     448,6963288768, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51436379 merged 5514570 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:47:47.832 INFO  [3599279] [QSim::simulate@479]   28 : sslice {     448,     464,7221188352, 257899584}257.899584
    2025-11-19 22:48:26.812 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   28 dt   38.972547 slice   28 : sslice {     448,     464,7221188352, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51435773 merged 5512953 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:48:27.028 INFO  [3599279] [QSim::simulate@479]   29 : sslice {     464,     480,7479087936, 257899584}257.899584
    2025-11-19 22:49:06.073 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   29 dt   39.038268 slice   29 : sslice {     464,     480,7479087936, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51435002 merged 5514453 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:49:06.304 INFO  [3599279] [QSim::simulate@479]   30 : sslice {     480,     496,7736987520, 257899584}257.899584
    2025-11-19 22:49:45.451 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   30 dt   39.140463 slice   30 : sslice {     480,     496,7736987520, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51435681 merged 5512837 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:49:45.691 INFO  [3599279] [QSim::simulate@479]   31 : sslice {     496,     512,7994887104, 257899584}257.899584
    2025-11-19 22:50:24.960 INFO  [3599279] [QSim::simulate@505]  eventID 0 xxl YES i   31 dt   39.262638 slice   31 : sslice {     496,     512,7994887104, 257899584}257.899584
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 257899584 selected 51436969 merged 5510811 selected/in   0.199 merged/selected   0.107 
    2025-11-19 22:50:28.433 INFO  [3599279] [QSim::simulate@535]  num_slice 32 has_hlm YES needs_final_merge YES
    ]SPM::merge_partial_select select_flagmask 8192 time_window   1.000 in 176424702 selected 176424702 merged 15394107 selected/in   1.000 merged/selected   0.087 
    2025-11-19 22:50:29.234 INFO  [3599279] [QSim::simulate_final_merge@640]  hlm (176424702, 4, ) fin (15394107, 4, )
    2025-11-19 22:50:29.243 INFO  [3599279] [QSim::simulate@550]  eventID 0 tot_dt 1251.720906 tot_ph 8252786688 tot_ph/M 8252.787109 tot_ht   15394107 tot_ht/M  15.394107 tot_ht/tot_ph   0.001865 reset_ YES
    2025-11-19 22:50:29.243 INFO  [3599279] [SEvt::save@4505] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvvlarge_evt_merge/A000 [genstep,hitlitemerged]
    2025-11-19 22:50:29.347 INFO  [3599279] [QSim::simulate@571] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 1263.469238 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 1259.328857 (multilaunch loop begin to end) 
     (PCAT - LEND)/M   4.035888 (topfold concat and clear subfold) 
     (TAIL - BRES)/M   0.102661 (QSim::reset which saves hits) 
     tot_idt/M       1251.721436 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          1251.720906 int(tot_dt*M)   1251720905 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M         7.378995 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-11-19 22:50:29.440  440813789 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 







TODO: Review device-side alloc, cleanup, especially with multi-launch running
--------------------------------------------------------------------------------------



CSGOptiX7.cu change to photon saving being optional::

    456     if( evt->photon )
    457     {
    458         evt->photon[idx] = ctx.p ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
    459     }
    460 
    461     if( evt->photonlite )
    462     {
    463         sphotonlite l ;
    464         l.init( ctx.p.identity, ctx.p.time, ctx.p.flagmask );
    465         l.set_lpos(prd->lposcost(), prd->lposfphi() );
    466         evt->photonlite[idx] = l ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
    467     }



mode_lite
----------

::

    (ok) A[blyth@localhost sysrap]$ opticks-f mode_lite
    ./CSGOptiX/cxs_min.sh:opticks_mode_lite=0
    ./CSGOptiX/cxs_min.sh:   opticks_mode_lite=2   # 0:Photon+Hit 1:PhotonLite+HitLite, 2:both (NB only controls Hit/Photon variants Lite-OR-not-OR-both, does not enable component)
    ./CSGOptiX/cxs_min.sh:   opticks_mode_lite=1          # 0:non-Lite 1:photon/hit -Lite variant  2:both non-Lite and Lite variants for debug
    ./CSGOptiX/cxs_min.sh:       opticks_mode_lite=1          # 0:non-Lite 1:photon/hit -Lite variant  2:both non-Lite and Lite variants for debug
    ./CSGOptiX/cxs_min.sh:export OPTICKS_MODE_LITE=${OPTICKS_MODE_LITE:-$opticks_mode_lite}
    ./cxs_min.sh:opticks_mode_lite=0
    ./cxs_min.sh:   opticks_mode_lite=2   # 0:Photon+Hit 1:PhotonLite+HitLite, 2:both (NB only controls Hit/Photon variants Lite-OR-not-OR-both, does not enable component)
    ./cxs_min.sh:   opticks_mode_lite=1          # 0:non-Lite 1:photon/hit -Lite variant  2:both non-Lite and Lite variants for debug
    ./cxs_min.sh:       opticks_mode_lite=1          # 0:non-Lite 1:photon/hit -Lite variant  2:both non-Lite and Lite variants for debug
    ./cxs_min.sh:export OPTICKS_MODE_LITE=${OPTICKS_MODE_LITE:-$opticks_mode_lite}
    ./qudarap/QEvt.cc:    evt->photonlite = evt->mode_lite > 0 ? QU::device_alloc_zero<sphotonlite>( evt->max_slot, "QEvt::device_alloc_photon/max_slot*sizeof(sphotonlite)" ) : nullptr ;
    ./sysrap/SEventConfig.hh:    static void SetModeLite(  int mode_lite);
    ./sysrap/sevent.h:    int   mode_lite    ; // 0 or 1 (also 2 when debug comparing)
    ./sysrap/sevent.h:    mode_lite     = SEventConfig::ModeLite() ;
    ./sysrap/sevent.h:        << " evt.mode_lite     " << std::setw(w) << mode_lite    << std::endl
    (ok) A[blyth@localhost opticks]$ 






sevent::mode_lite generalization is needed
---------------------------------------------

::

    1331 void QEvt::setNumPhoton(unsigned num_photon )
    1332 {
    1333     LOG_IF(info, LIFECYCLE) << " num_photon " << num_photon ;
    1334     LOG(LEVEL);
    1335 
    1336     sev->setNumPhoton(num_photon);
    1337     if( evt->photon == nullptr ) device_alloc_photon();
    1338     uploadEvt();
    1339 }


    1363 void QEvt::device_alloc_photon()
    1364 {
    1365     LOG_IF(info, LIFECYCLE) ;
    1366     SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
    1367 
    1368     LOG(LEVEL)
    1369         << " evt.max_slot   " << evt->max_slot
    1370         << " evt.max_record " << evt->max_record
    1371         << " evt.max_photon " << evt->max_photon
    1372         << " evt.num_photon " << evt->num_photon
    1373 #ifndef PRODUCTION
    1374         << " evt.num_record " << evt->num_record
    1375         << " evt.num_rec    " << evt->num_rec
    1376         << " evt.num_seq    " << evt->num_seq
    1377         << " evt.num_prd    " << evt->num_prd
    1378         << " evt.num_tag    " << evt->num_tag
    1379         << " evt.num_flat   " << evt->num_flat
    1380 #endif
    1381         ;
    1382 
    1383     evt->photon  = evt->max_slot > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot, "QEvt::device_alloc_photon/max_slot*sizeof(sphoton)" ) : nullptr ;
    1384     evt->photonlite = evt->mode_lite > 0 ? QU::device_alloc_zero<sphotonlite>( evt->max_slot, "QEvt::device_alloc_photon/max_slot*sizeof(sphotonlite)" ) : nullptr ;
    1385 
    1386 #ifndef PRODUCTION
    1387     evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot * evt->max_record, "max_slot*max_record*sizeof(sphoton)" ) : nullptr ;
    1388     evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_slot * evt->max_rec   , "max_slot*max_rec*sizeof(srec)"    ) : nullptr ;
    1389     evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_slot * evt->max_prd   , "max_slot*max_prd*sizeof(quad2)"    ) : nullptr ;
    1390     evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_slot                  , "max_slot*sizeof(sseq)"    ) : nullptr ;
    1391     evt->tag     = evt->max_tag   == 1 ? QU::device_alloc_zero<stag>(    evt->max_slot                  , "max_slot*sizeof(stag)"    ) : nullptr ;
    1392     evt->flat    = evt->max_flat  == 1 ? QU::device_alloc_zero<sflat>(   evt->max_slot                  , "max_slot*sizeof(sflat)"   ) : nullptr ;
    1393 #endif
    1394 
    1395     LOG(LEVEL) << desc() ;
    1396     LOG(LEVEL) << desc_alloc() ;
    1397 }



Potential smoking gun for leak : the latch not handling photonlite : repeated QEvt::device_alloc_photon
----------------------------------------------------------------------------------------------------------

::

    num_slice 32
    2025-11-19 21:03:14.432 INFO  [3568158] [QSim::simulate@479]    0 : sslice {       0,      16,         0, 257899584}257.899584

    Thread 1 "CSGOptiXSMTest" hit Breakpoint 1, QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1370
    1370	    LOG_IF(info, LIFECYCLE) ;
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libnvidia-gpucomp-580.82.07-1.el9.x86_64 libnvidia-ml-580.82.07-1.el9.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64 nvidia-driver-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1370
    #1  0x00007ffff5ec1a37 in QEvt::setNumPhoton (this=0x186569c0, num_photon=257899584) at /home/blyth/opticks/qudarap/QEvt.cc:1337
    #2  0x00007ffff5ebb428 in QEvt::setGenstepUpload (this=0x186569c0, qq0=0x1dba8700, gs_start=0, gs_stop=16) at /home/blyth/opticks/qudarap/QEvt.cc:426
    #3  0x00007ffff5eb9efa in QEvt::setGenstepUpload_NP (this=0x186569c0, gs_=0x1e6a8780, gss_=0x1e6262e0) at /home/blyth/opticks/qudarap/QEvt.cc:225
    #4  0x00007ffff5e716c9 in QSim::simulate (this=0x18656a20, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:481
    #5  0x00007ffff7e35b54 in CSGOptiX::simulate (this=0x1866b3e0, eventID=0, reset=true) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:777
    #6  0x00007ffff7e3255c in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:177
    #7  0x0000000000404a95 in main (argc=1, argv=0x7fffffffaf28) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) c
    Continuing.
    2025-11-19 21:14:45.351 INFO  [3568158] [QSim::simulate@505]  eventID 0 xxl YES i    0 dt   38.198782 slice    0 : sslice {       0,      16,         0, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51442248 merged 5511645 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:14:45.590 INFO  [3568158] [QSim::simulate@479]    1 : sslice {      16,      32, 257899584, 257899584}257.899584

    Thread 1 "CSGOptiXSMTest" hit Breakpoint 1, QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1370
    1370	    LOG_IF(info, LIFECYCLE) ;
    (gdb) bt
    #0  QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1370
    #1  0x00007ffff5ec1a37 in QEvt::setNumPhoton (this=0x186569c0, num_photon=257899584) at /home/blyth/opticks/qudarap/QEvt.cc:1337
    #2  0x00007ffff5ebb428 in QEvt::setGenstepUpload (this=0x186569c0, qq0=0x1dba8700, gs_start=16, gs_stop=32) at /home/blyth/opticks/qudarap/QEvt.cc:426
    #3  0x00007ffff5eb9efa in QEvt::setGenstepUpload_NP (this=0x186569c0, gs_=0x1e6a8780, gss_=0x1e626300) at /home/blyth/opticks/qudarap/QEvt.cc:225
    #4  0x00007ffff5e716c9 in QSim::simulate (this=0x18656a20, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:481
    #5  0x00007ffff7e35b54 in CSGOptiX::simulate (this=0x1866b3e0, eventID=0, reset=true) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:777
    #6  0x00007ffff7e3255c in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:177
    #7  0x0000000000404a95 in main (argc=1, argv=0x7fffffffaf28) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) c
    Continuing.
    2025-11-19 21:17:55.950 INFO  [3568158] [QSim::simulate@505]  eventID 0 xxl YES i    1 dt   38.255218 slice    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51430837 merged 5514117 selected/in   0.199 merged/selected   0.107 
    2025-11-19 21:17:56.172 INFO  [3568158] [QSim::simulate@479]    2 : sslice {      32,      48, 515799168, 257899584}257.899584

    Thread 1 "CSGOptiXSMTest" hit Breakpoint 1, QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1370
    1370	    LOG_IF(info, LIFECYCLE) ;
    (gdb) bt
    #0  QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1370
    #1  0x00007ffff5ec1a37 in QEvt::setNumPhoton (this=0x186569c0, num_photon=257899584) at /home/blyth/opticks/qudarap/QEvt.cc:1337
    #2  0x00007ffff5ebb428 in QEvt::setGenstepUpload (this=0x186569c0, qq0=0x1dba8700, gs_start=32, gs_stop=48) at /home/blyth/opticks/qudarap/QEvt.cc:426
    #3  0x00007ffff5eb9efa in QEvt::setGenstepUpload_NP (this=0x186569c0, gs_=0x1e6a8780, gss_=0x1e626320) at /home/blyth/opticks/qudarap/QEvt.cc:225
    #4  0x00007ffff5e716c9 in QSim::simulate (this=0x18656a20, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:481
    #5  0x00007ffff7e35b54 in CSGOptiX::simulate (this=0x1866b3e0, eventID=0, reset=true) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:777
    #6  0x00007ffff7e3255c in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:177
    #7  0x0000000000404a95 in main (argc=1, argv=0x7fffffffaf28) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 




After fix the latch
-----------------------


::

    (gdb) bt
    #0  QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1388
    #1  0x00007ffff5ec1a3c in QEvt::setNumPhoton (this=0x186569c0, num_photon=257899584) at /home/blyth/opticks/qudarap/QEvt.cc:1339
    #2  0x00007ffff5ebb428 in QEvt::setGenstepUpload (this=0x186569c0, qq0=0x1bc53340, gs_start=0, gs_stop=16) at /home/blyth/opticks/qudarap/QEvt.cc:426
    #3  0x00007ffff5eb9efa in QEvt::setGenstepUpload_NP (this=0x186569c0, gs_=0x1ae45e70, gss_=0x1afe6650) at /home/blyth/opticks/qudarap/QEvt.cc:225
    #4  0x00007ffff5e716c9 in QSim::simulate (this=0x18656a20, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:481
    #5  0x00007ffff7e35b54 in CSGOptiX::simulate (this=0x1866b3e0, eventID=0, reset=true) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:777
    #6  0x00007ffff7e3255c in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:177
    #7  0x0000000000404a95 in main (argc=1, argv=0x7fffffffaf28) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) f 0
    #0  QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1388
    1388	{
    (gdb) list
    1383	
    1384	
    1385	**/
    1386	
    1387	void QEvt::device_alloc_photon()
    1388	{
    1389	    LOG_IF(info, LIFECYCLE) ;
    1390	    SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
    1391	
    1392	    bool with_photon     = evt->with_photon(); 
    (gdb) b 1424
    Breakpoint 2 at 0x7ffff5ec2185: file /home/blyth/opticks/qudarap/QEvt.cc, line 1424.
    (gdb) c
    Continuing.

    Thread 1 "CSGOptiXSMTest" hit Breakpoint 2, QEvt::device_alloc_photon (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1424
    1424	    evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_slot                  , "max_slot*sizeof(sseq)"    ) : nullptr ;
    (gdb) printf "%s\n", evt->descBuf().c_str()
        sevent::descBuf 
        evt.genstep         Y       0x7fff1c000000        num_genstep      16        max_genstep 10000000
             evt.seed       Y       0x7ffedc000000           num_seed 257899584         max_photon 1000000000000
             evt.photon     N                    0         num_photon 257899584         max_photon 1000000000000
         evt.photonlite     Y       0x7ffde2000000     num_photonlite 257899584         max_photon 1000000000000
             evt.record     N                    0         num_record       0         max_record       0
                evt.rec     N                    0            num_rec       0            max_rec       0
                evt.aux     N                    0            num_aux       0            max_aux       0
                evt.sup     N                    0            num_sup       0            max_sup       0
                evt.seq     N                    0            num_seq       0            max_seq       0
                evt.hit     N                    0            num_hit       0         max_photon 1000000000000
           evt.simtrace     N                    0       num_simtrace       0       max_simtrace 1000000000000
                evt.prd     N                    0            num_prd       0            max_prd       0
                evt.tag     N                    0            num_tag       0            max_tag       0
               evt.flat     N                    0           num_flat       0           max_flat       0

    (gdb) 











::

    #22 0x00007ffff5773e0e in thrust::THRUST_200302_700_890_NS::sort_by_key<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite> > (exec=..., keys_first=..., keys_last=..., values_first=...) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/sort.inl:102
    #23 0x00007ffff576cc7f in SPM::merge_partial_select (d_in=0x7ff9fa000000, num_in=257899584, d_out=0x166c7a90, num_out=0x166c7a08, select_flagmask=8192, time_window=1, stream=0x0) at /home/blyth/opticks/sysrap/SPM.cu:266
    #24 0x00007ffff5ec0e61 in QEvt::PerLaunchMerge (evt=0x166c7960, stream=0x0) at /home/blyth/opticks/qudarap/QEvt.cc:1112
    #25 0x00007ffff5ec03fd in QEvt::gatherHitLiteMerged (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1033
    #26 0x00007ffff5ec17a1 in QEvt::gatherComponent_ (this=0x186569c0, cmp=67108864) at /home/blyth/opticks/qudarap/QEvt.cc:1295
    #27 0x00007ffff5ec142f in QEvt::gatherComponent (this=0x186569c0, cmp=67108864) at /home/blyth/opticks/qudarap/QEvt.cc:1270
    #28 0x00007ffff57baf39 in SEvt::gather_components (this=0x160a72c0) at /home/blyth/opticks/sysrap/SEvt.cc:3929
    #29 0x00007ffff57bbb4c in SEvt::gather (this=0x160a72c0) at /home/blyth/opticks/sysrap/SEvt.cc:4035
    #30 0x00007ffff5e71d53 in QSim::simulate (this=0x18656a20, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:515
    #31 0x00007ffff7e35b54 in CSGOptiX::simulate (this=0x1866b3e0, eventID=0, reset=true) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:777
    #32 0x00007ffff7e3255c in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:177
    #33 0x0000000000404a95 in main (argc=1, argv=0x7fffffffaf48) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) f 24
    #24 0x00007ffff5ec0e61 in QEvt::PerLaunchMerge (evt=0x166c7960, stream=0x0) at /home/blyth/opticks/qudarap/QEvt.cc:1112
    1112	    SPM::merge_partial_select(
    (gdb) list
    1107	
    1108	
    1109	
    1110	NP* QEvt::PerLaunchMerge(sevent* evt, cudaStream_t stream ) // static
    1111	{
    1112	    SPM::merge_partial_select(
    1113	         evt->photonlite,
    1114	         evt->num_photonlite,
    1115	         &evt->hitlitemerged,
    1116	         &evt->num_hitlitemerged,
    (gdb) printf "%s\n", evt->descBuf().c_str()
        sevent::descBuf 
        evt.genstep         Y       0x7fff1c000000        num_genstep      16        max_genstep 10000000
             evt.seed       Y       0x7ffedc000000           num_seed 257899584         max_photon 1000000000000
             evt.photon     Y       0x7ffaf4000000         num_photon 257899584         max_photon 1000000000000
         evt.photonlite     Y       0x7ff9fa000000     num_photonlite 257899584         max_photon 1000000000000
             evt.record     N                    0         num_record       0         max_record       0
                evt.rec     N                    0            num_rec       0            max_rec       0
                evt.aux     N                    0            num_aux       0            max_aux       0
                evt.sup     N                    0            num_sup       0            max_sup       0
                evt.seq     N                    0            num_seq       0            max_seq       0
                evt.hit     N                    0            num_hit       0         max_photon 1000000000000
           evt.simtrace     N                    0       num_simtrace       0       max_simtrace 1000000000000
                evt.prd     N                    0            num_prd       0            max_prd       0
                evt.tag     N                    0            num_tag       0            max_tag       0
               evt.flat     N                    0           num_flat       0           max_flat       0

    (gdb) 



BASELINE : TEST=vvvvvlarge_evt
-------------------------------

::

    (ok) A[blyth@localhost CSGOptiX]$ cxs_min.sh run

    WARNING : DEBUG RUNNING WITH OPTICKS_EVENT_MODE Hit IS APPROPRIATE FOR SMALL STATISTICS ONLY

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
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2025-11-19 19:00:32.785  785034244 : [/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    2025-11-19 19:00:34.470 INFO  [3541812] [SEventConfig::SetDevice@1785] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262326496
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 0

    2025-11-19 19:00:34.470 INFO  [3541812] [SEventConfig::SetDevice@1797]  Configured_MaxSlot/M 0 Final_MaxSlot/M 262 HeuristicMaxSlot_Rounded/M 262 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2025-11-19 19:00:34.602 INFO  [3541812] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-19 19:00:34.711 INFO  [3541812] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000000 SEventConfig::MaxCurand 1000000000000
    2025-11-19 19:00:34.711 INFO  [3541812] [QRng::init@104] [QRng__init_VERBOSE] YES
    QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 1000000000000
                         rngmax/M 1000000
                               qr 0x19758360
        qr.skipahead_event_offset 100000
                             d_qr 0x7fe8cec00200

    SPMT::init_pmtNum pmtNum.sstr (6, )
    [SPMT::init_pmtNum
    [SPMT_Num.desc
    m_nums_cd_lpmt     :  17612
    m_nums_cd_spmt     :  25600
    m_nums_wp_lpmt     :   2400
    m_nums_wp_atm_lpmt :    348
    m_nums_wp_wal_pmt  :      5
    m_nums_wp_atm_mpmt :    600
    ALL                :  46565
    SUM                :  46565
    SUM==ALL           :    YES
    ]SPMT_Num.desc
    ]SPMT::init_pmtNum
    2025-11-19 19:00:35.054 INFO  [3541812] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-19 19:00:35.056 INFO  [3541812] [QSim::simulate@457]  eventID      0 igs (512, 6, 4, ) tot_ph_0 8252786688 tot_ph_0/M 8252 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 32 TotalPhoton 8252786688 TotalPhoton/M 8252.786688
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      16,         0, 257899584}257.899584
       1 : sslice {      16,      32, 257899584, 257899584}257.899584
       2 : sslice {      32,      48, 515799168, 257899584}257.899584
       3 : sslice {      48,      64, 773698752, 257899584}257.899584
       4 : sslice {      64,      80,1031598336, 257899584}257.899584
       5 : sslice {      80,      96,1289497920, 257899584}257.899584
       6 : sslice {      96,     112,1547397504, 257899584}257.899584
       7 : sslice {     112,     128,1805297088, 257899584}257.899584
       8 : sslice {     128,     144,2063196672, 257899584}257.899584
       9 : sslice {     144,     160,2321096256, 257899584}257.899584
      10 : sslice {     160,     176,2578995840, 257899584}257.899584
      11 : sslice {     176,     192,2836895424, 257899584}257.899584
      12 : sslice {     192,     208,3094795008, 257899584}257.899584
      13 : sslice {     208,     224,3352694592, 257899584}257.899584
      14 : sslice {     224,     240,3610594176, 257899584}257.899584
      15 : sslice {     240,     256,3868493760, 257899584}257.899584
      16 : sslice {     256,     272,4126393344, 257899584}257.899584
      17 : sslice {     272,     288,4384292928, 257899584}257.899584
      18 : sslice {     288,     304,4642192512, 257899584}257.899584
      19 : sslice {     304,     320,4900092096, 257899584}257.899584
      20 : sslice {     320,     336,5157991680, 257899584}257.899584
      21 : sslice {     336,     352,5415891264, 257899584}257.899584
      22 : sslice {     352,     368,5673790848, 257899584}257.899584
      23 : sslice {     368,     384,5931690432, 257899584}257.899584
      24 : sslice {     384,     400,6189590016, 257899584}257.899584
      25 : sslice {     400,     416,6447489600, 257899584}257.899584
      26 : sslice {     416,     432,6705389184, 257899584}257.899584
      27 : sslice {     432,     448,6963288768, 257899584}257.899584
      28 : sslice {     448,     464,7221188352, 257899584}257.899584
      29 : sslice {     464,     480,7479087936, 257899584}257.899584
      30 : sslice {     480,     496,7736987520, 257899584}257.899584
      31 : sslice {     496,     512,7994887104, 257899584}257.899584
                      start    stop     offset      count    count/M 
     num_slice 32
    2025-11-19 19:01:12.807 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    0 dt   37.709253 slice    0 : sslice {       0,      16,         0, 257899584}257.899584
    2025-11-19 19:01:54.538 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    1 dt   38.392592 slice    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    2025-11-19 19:02:36.611 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    2 dt   38.771796 slice    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    2025-11-19 19:03:18.758 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    3 dt   38.884052 slice    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    2025-11-19 19:04:01.322 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    4 dt   38.917745 slice    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    2025-11-19 19:04:43.885 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    5 dt   38.902384 slice    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    2025-11-19 19:05:26.477 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    6 dt   38.883522 slice    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    2025-11-19 19:06:09.194 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    7 dt   38.977103 slice    7 : sslice {     112,     128,1805297088, 257899584}257.899584
    2025-11-19 19:06:51.675 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    8 dt   38.876769 slice    8 : sslice {     128,     144,2063196672, 257899584}257.899584
    2025-11-19 19:07:34.248 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i    9 dt   38.884377 slice    9 : sslice {     144,     160,2321096256, 257899584}257.899584
    2025-11-19 19:08:16.831 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   10 dt   38.876392 slice   10 : sslice {     160,     176,2578995840, 257899584}257.899584
    2025-11-19 19:08:59.376 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   11 dt   38.852607 slice   11 : sslice {     176,     192,2836895424, 257899584}257.899584
    2025-11-19 19:09:42.064 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   12 dt   38.868976 slice   12 : sslice {     192,     208,3094795008, 257899584}257.899584
    2025-11-19 19:10:24.637 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   13 dt   38.893854 slice   13 : sslice {     208,     224,3352694592, 257899584}257.899584
    2025-11-19 19:11:07.149 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   14 dt   38.892303 slice   14 : sslice {     224,     240,3610594176, 257899584}257.899584
    2025-11-19 19:11:49.731 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   15 dt   38.881760 slice   15 : sslice {     240,     256,3868493760, 257899584}257.899584
    2025-11-19 19:12:32.267 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   16 dt   38.875539 slice   16 : sslice {     256,     272,4126393344, 257899584}257.899584
    2025-11-19 19:13:15.014 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   17 dt   38.871989 slice   17 : sslice {     272,     288,4384292928, 257899584}257.899584
    2025-11-19 19:13:57.650 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   18 dt   38.881935 slice   18 : sslice {     288,     304,4642192512, 257899584}257.899584
    2025-11-19 19:14:40.060 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   19 dt   38.897351 slice   19 : sslice {     304,     320,4900092096, 257899584}257.899584
    2025-11-19 19:15:22.714 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   20 dt   38.880945 slice   20 : sslice {     320,     336,5157991680, 257899584}257.899584
    2025-11-19 19:16:05.430 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   21 dt   38.900078 slice   21 : sslice {     336,     352,5415891264, 257899584}257.899584
    2025-11-19 19:16:47.998 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   22 dt   38.891960 slice   22 : sslice {     352,     368,5673790848, 257899584}257.899584
    2025-11-19 19:17:33.567 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   23 dt   38.858991 slice   23 : sslice {     368,     384,5931690432, 257899584}257.899584
    2025-11-19 19:18:17.066 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   24 dt   38.910510 slice   24 : sslice {     384,     400,6189590016, 257899584}257.899584
    2025-11-19 19:18:59.495 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   25 dt   38.934993 slice   25 : sslice {     400,     416,6447489600, 257899584}257.899584
    2025-11-19 19:19:42.111 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   26 dt   38.913713 slice   26 : sslice {     416,     432,6705389184, 257899584}257.899584
    2025-11-19 19:20:24.776 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   27 dt   38.914959 slice   27 : sslice {     432,     448,6963288768, 257899584}257.899584
    2025-11-19 19:21:07.524 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   28 dt   38.892848 slice   28 : sslice {     448,     464,7221188352, 257899584}257.899584
    2025-11-19 19:21:50.140 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   29 dt   38.914191 slice   29 : sslice {     464,     480,7479087936, 257899584}257.899584
    2025-11-19 19:22:32.802 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   30 dt   38.895630 slice   30 : sslice {     480,     496,7736987520, 257899584}257.899584
    2025-11-19 19:23:15.487 INFO  [3541812] [QSim::simulate@505]  eventID 0 xxl YES i   31 dt   38.907571 slice   31 : sslice {     496,     512,7994887104, 257899584}257.899584
    2025-11-19 19:29:56.871 INFO  [3541812] [QSim::simulate@550]  eventID 0 tot_dt 1242.808689 tot_ph 8252786688 tot_ph/M 8252.787109 tot_ht 1646084065 tot_ht/M 1646.084106 tot_ht/tot_ph   0.199458 reset_ YES
    2025-11-19 19:29:56.896 INFO  [3541812] [SEvt::save@4505] /data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvvlarge_evt/A000 [genstep,hit]
    2025-11-19 19:37:53.717 INFO  [3541812] [QSim::simulate@571] 
    SEvt__MINTIME
     (TAIL - HEAD)/M 2238.661865 (head to tail of QSim::simulate method) 
     (LEND - LBEG)/M 1364.171021 (multilaunch loop begin to end) 
     (PCAT - LEND)/M 397.591064 (topfold concat and clear subfold) 
     (TAIL - BRES)/M 476.844727 (QSim::reset which saves hits) 
     tot_idt/M       1242.809204 (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt          1242.808689 int(tot_dt*M)   1242808689 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 
     tot_gdt/M       121.056656 (sum of SEvt::gather int64_t stamp differences in microseconds)

    2025-11-19 19:37:55.463  463093415 : ]/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    (ok) A[blyth@localhost CSGOptiX]$ 





TEST=vvvvvlarge_evt_merge cxs_min.sh run
------------------------------------------


::


    (ok) A[blyth@localhost CSGOptiX]$ cxs_min.sh run

    WARNING : DEBUG RUNNING WITH OPTICKS_EVENT_MODE Hit IS APPROPRIATE FOR SMALL STATISTICS ONLY

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
    2025-11-19 19:56:40.973  973583656 : [/data1/blyth/local/opticks_Debug/bin/cxs_min.sh 
    SLOG::EnvLevel adjusting loglevel by envvar   key QSim level INFO fallback DEBUG upper_level INFO
    2025-11-19 19:56:44.290 INFO  [3546890] [SEventConfig::SetDevice@1785] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262326496
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 0

    2025-11-19 19:56:44.290 INFO  [3546890] [SEventConfig::SetDevice@1797]  Configured_MaxSlot/M 0 Final_MaxSlot/M 262 HeuristicMaxSlot_Rounded/M 262 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2025-11-19 19:56:44.452 INFO  [3546890] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-19 19:56:44.461 INFO  [3546890] [QSim::UploadComponents@118] [ ssim 0xd84990
    2025-11-19 19:56:44.461 INFO  [3546890] [QSim::UploadComponents@121] [ new QBase
    2025-11-19 19:56:44.573 INFO  [3546890] [QSim::UploadComponents@123] ] new QBase : latency here of about 0.3s from first device access, if latency of >1s need to start nvidia-persistenced 
    2025-11-19 19:56:44.573 INFO  [3546890] [QSim::UploadComponents@124] QBase::desc base 0x17027dc0 d_base 0x7f034ac00000 base.desc qbase::desc pidx 1099511627775
    2025-11-19 19:56:44.573 INFO  [3546890] [QSim::UploadComponents@128] [ new QRng skipahead_event_offset : 100000 OPTICKS_EVENT_SKIPAHEAD
    2025-11-19 19:56:44.573 INFO  [3546890] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000000 SEventConfig::MaxCurand 1000000000000
    2025-11-19 19:56:44.573 INFO  [3546890] [QRng::init@104] [QRng__init_VERBOSE] YES
    QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 1000000000000
                         rngmax/M 1000000
                               qr 0x189cc6e0
        qr.skipahead_event_offset 100000
                             d_qr 0x7f034ac00200

    2025-11-19 19:56:44.573 INFO  [3546890] [QSim::UploadComponents@130] ] new QRng QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 1000000000000
                         rngmax/M 1000000
                               qr 0x189cc6e0
        qr.skipahead_event_offset 100000
                             d_qr 0x7f034ac00200

    2025-11-19 19:56:44.573 INFO  [3546890] [QSim::UploadComponents@132] QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 1000000000000
                         rngmax/M 1000000
                               qr 0x189cc6e0
        qr.skipahead_event_offset 100000
                             d_qr 0x7f034ac00200

    2025-11-19 19:56:44.577 INFO  [3546890] [QSim::UploadComponents@145] QOptical optical NP  dtype <i4(324, 4, 4, ) size 5184 uifc i ebyte 4 shape.size 3 data.size 20736 meta.size 0 names.size 0
    2025-11-19 19:56:44.623 INFO  [3546890] [QSim::UploadComponents@148] QBnd src NP  dtype <f4(324, 4, 2, 761, 4, ) size 7890048 uifc f ebyte 4 shape.size 5 data.size 31560192 meta.size 61 names.size 324 tex QTex width 761 height 2592 texObj 1 meta 0x189e20a0 d_meta 0x7f034ac05600 tex 0x189e21c0
    2025-11-19 19:56:44.623 INFO  [3546890] [QSim::UploadComponents@152] QDebug::desc  dbg 0x189d9bd0 d_dbg 0x7f034ac05a00
     QState::Desc QState::Desc
    material1 ( 1.000,1000.000,1000.000, 0.000) 
    material2 ( 1.500,1000.000,1000.000, 0.000) 
    m1group2  (300.000, 0.000, 0.000, 0.000) 
    surface   ( 0.000, 0.000, 0.000, 0.000) 
    optical   (     0,     0,     0,     0) 

     dbg.p.desc  pos ( 0.000, 0.000, 0.000)  t     0.000  mom ( 1.000, 0.000, 0.000)  iindex 1065353216  pol ( 0.000, 1.000, 0.000)  wl  500.000   bn 0 fl 0 id 0 or 1 ix 0 fm 0 ab    ii 1065353216
    2025-11-19 19:56:44.625 INFO  [3546890] [QSim::UploadComponents@165]   propcom null, snam::PROPCOM propcom.npy
    2025-11-19 19:56:44.626 INFO  [3546890] [QSim::UploadComponents@178] QScint dsrc NP  dtype <f8(3, 4096, 1, ) size 12288 uifc f ebyte 8 shape.size 3 data.size 98304 meta.size 97 names.size 1 src NP  dtype <f4(3, 4096, 1, ) size 12288 uifc f ebyte 4 shape.size 3 data.size 49152 meta.size 97 names.size 1 tex QTex width 4096 height 3 texObj 2 meta 0x189d8660 d_meta 0x7f034ac05c00 tex 0x189e53a0
    2025-11-19 19:56:44.626 INFO  [3546890] [QSim::UploadComponents@187] QCerenkov fold - icdf_ - icdf - tex 0
    SPMT::init_pmtNum pmtNum.sstr (6, )
    [SPMT::init_pmtNum
    [SPMT_Num.desc
    m_nums_cd_lpmt     :  17612
    m_nums_cd_spmt     :  25600
    m_nums_wp_lpmt     :   2400
    m_nums_wp_atm_lpmt :    348
    m_nums_wp_wal_pmt  :      5
    m_nums_wp_atm_mpmt :    600
    ALL                :  46565
    SUM                :  46565
    SUM==ALL           :    YES
    ]SPMT_Num.desc
    ]SPMT::init_pmtNum
    2025-11-19 19:56:44.637 INFO  [3546890] [QSim::UploadComponents@217] QPMT<float> WITH_CUSTOM4  INSTANCE:YES QPMT::desc
                           rindex (24, 15, 2, )
                          qeshape (3, 44, 2, )
                          cetheta (3, 10, 2, )
                          cecosth (3, 10, 2, )
                        thickness (3, 4, 1, )
                             lcqs (20965, 2, )
                        s_qeshape (1, 61, 2, )
                        s_qescale (25600, 1, )
                  pmt.rindex_prop 0x7f034ac06e00
                 pmt.qeshape_prop 0x7f034ac07600
                 pmt.cetheta_prop 0x7f034ac07a00
                 pmt.cecosth_prop 0x7f034ac07e00
                    pmt.thickness 0x7f034ac08400
                         pmt.lcqs 0x7f034ac08600
                            d_pmt 0x7f034ac4a600

     spmt_f YES qpmt YES
    2025-11-19 19:56:44.637 INFO  [3546890] [QSim::UploadComponents@229]  multifilm null, snam::MULTIFILM multifilm.npy
    2025-11-19 19:56:44.637 INFO  [3546890] [QSim::UploadComponents@236] ] ssim 0xd84990
    2025-11-19 19:56:44.637 INFO  [3546890] [QSim::QSim@280] QSim::desc
     this 0x18f9dc20 INSTANCE 0x0 QEvt.hh:qev 0x18f9dbc0 qsim.h:sim 0x0
    2025-11-19 19:56:44.638 INFO  [3546890] [QSim::init@324]  MISSING_PMT NO  has_PMT YES QSim::pmt YES QSim::pmt->d_pmt YES [QSim__REQUIRE_PMT] NO 
    2025-11-19 19:56:44.638 INFO  [3546890] [QSim::init@347] QSim::desc
     this 0x18f9dc20 INSTANCE 0x18f9dc20 QEvt.hh:qev 0x18f9dbc0 qsim.h:sim 0x18f9b890
    2025-11-19 19:56:44.638 INFO  [3546890] [QSim::init@348] 
    QSim::descComponents
     (QBase)base             YES
     (QEvt)qev           YES
     (SEvt)sev               YES
     (QRng)rng               YES
     (QScint)scint           YES
     (QCerenkov)cerenkov     YES
     (QBnd)bnd               YES
     (QOptical)optical       NO 
     (QDebug)debug_          YES
     (QProp)prop             YES
     (QPMT)pmt               YES
     (QMultiFilm)multifilm   NO 
     (qsim)sim               YES
     (qsim)d_sim             YES
     (qdebug)dbg             YES
     (qdebug)d_dbg           YES

    2025-11-19 19:56:45.036 INFO  [3546890] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-19 19:56:45.038 INFO  [3546890] [QSim::MaybeSaveIGS@729]  eventID 0 igs (512, 6, 4, ) igs_null NO  [QSim__SAVE_IGS_EVENTID] -1 [QSim__SAVE_IGS_PATH] $TMP/.opticks/igs.npy igs_path [/data1/blyth/tmp/.opticks/igs.npy] save_igs NO 
    2025-11-19 19:56:45.038 INFO  [3546890] [QSim::simulate@457]  eventID      0 igs (512, 6, 4, ) tot_ph_0 8252786688 tot_ph_0/M 8252 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 32 TotalPhoton 8252786688 TotalPhoton/M 8252.786688
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      16,         0, 257899584}257.899584
       1 : sslice {      16,      32, 257899584, 257899584}257.899584
       2 : sslice {      32,      48, 515799168, 257899584}257.899584
       3 : sslice {      48,      64, 773698752, 257899584}257.899584
       4 : sslice {      64,      80,1031598336, 257899584}257.899584
       5 : sslice {      80,      96,1289497920, 257899584}257.899584
       6 : sslice {      96,     112,1547397504, 257899584}257.899584
       7 : sslice {     112,     128,1805297088, 257899584}257.899584
       8 : sslice {     128,     144,2063196672, 257899584}257.899584
       9 : sslice {     144,     160,2321096256, 257899584}257.899584
      10 : sslice {     160,     176,2578995840, 257899584}257.899584
      11 : sslice {     176,     192,2836895424, 257899584}257.899584
      12 : sslice {     192,     208,3094795008, 257899584}257.899584
      13 : sslice {     208,     224,3352694592, 257899584}257.899584
      14 : sslice {     224,     240,3610594176, 257899584}257.899584
      15 : sslice {     240,     256,3868493760, 257899584}257.899584
      16 : sslice {     256,     272,4126393344, 257899584}257.899584
      17 : sslice {     272,     288,4384292928, 257899584}257.899584
      18 : sslice {     288,     304,4642192512, 257899584}257.899584
      19 : sslice {     304,     320,4900092096, 257899584}257.899584
      20 : sslice {     320,     336,5157991680, 257899584}257.899584
      21 : sslice {     336,     352,5415891264, 257899584}257.899584
      22 : sslice {     352,     368,5673790848, 257899584}257.899584
      23 : sslice {     368,     384,5931690432, 257899584}257.899584
      24 : sslice {     384,     400,6189590016, 257899584}257.899584
      25 : sslice {     400,     416,6447489600, 257899584}257.899584
      26 : sslice {     416,     432,6705389184, 257899584}257.899584
      27 : sslice {     432,     448,6963288768, 257899584}257.899584
      28 : sslice {     448,     464,7221188352, 257899584}257.899584
      29 : sslice {     464,     480,7479087936, 257899584}257.899584
      30 : sslice {     480,     496,7736987520, 257899584}257.899584
      31 : sslice {     496,     512,7994887104, 257899584}257.899584
                      start    stop     offset      count    count/M 
     num_slice 32
    2025-11-19 19:56:45.038 INFO  [3546890] [QSim::simulate@479]    0 : sslice {       0,      16,         0, 257899584}257.899584
    2025-11-19 19:57:22.729 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    0 dt   37.634852 slice    0 : sslice {       0,      16,         0, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 num_selected 51442248 merged 5511645 
    2025-11-19 19:57:22.993 INFO  [3546890] [QSim::simulate@479]    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    2025-11-19 19:58:01.560 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    1 dt   38.560736 slice    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 num_selected 51430837 merged 5514117 
    2025-11-19 19:58:01.794 INFO  [3546890] [QSim::simulate@479]    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    2025-11-19 19:58:40.528 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    2 dt   38.726058 slice    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 num_selected 51433012 merged 5515333 
    2025-11-19 19:58:40.753 INFO  [3546890] [QSim::simulate@479]    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    2025-11-19 19:59:19.800 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    3 dt   39.040124 slice    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 num_selected 51449616 merged 5514339 
    2025-11-19 19:59:20.038 INFO  [3546890] [QSim::simulate@479]    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    2025-11-19 19:59:59.175 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    4 dt   39.130745 slice    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 num_selected 51435671 merged 5512175 
    2025-11-19 19:59:59.402 INFO  [3546890] [QSim::simulate@479]    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    2025-11-19 20:00:38.574 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    5 dt   39.165270 slice    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 num_selected 51441825 merged 5513278 
    2025-11-19 20:00:38.797 INFO  [3546890] [QSim::simulate@479]    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    2025-11-19 20:01:17.937 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    6 dt   39.132556 slice    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 num_selected 51444895 merged 5513233 
    2025-11-19 20:01:18.182 INFO  [3546890] [QSim::simulate@479]    7 : sslice {     112,     128,1805297088, 257899584}257.899584
    2025-11-19 20:01:57.232 INFO  [3546890] [QSim::simulate@505]  eventID 0 xxl YES i    7 dt   39.043767 slice    7 : sslice {     112,     128,1805297088, 257899584}257.899584
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh: line 793: 3546890 Aborted                 (core dumped) $bin
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh run error
    (ok) A[blyth@localhost CSGOptiX]$ 




TEST=vvvvvlarge_evt_merge cxs_min.sh dbg
------------------------------------------

* nvtop : during running shows several VRAM steps before the abort

::



    2025-11-19 20:07:54.712 INFO  [3551660] [SSim::AnnotateFrame@197]  caller CSGFoundry::getFrameE tree YES elv NO  extra.size 0 tree_digest 7f69f317d34b7af58b4a07460ef20d39 dynamic 7f69f317d34b7af58b4a07460ef20d39
    2025-11-19 20:07:54.750 INFO  [3551660] [QSim::MaybeSaveIGS@729]  eventID 0 igs (512, 6, 4, ) igs_null NO  [QSim__SAVE_IGS_EVENTID] -1 [QSim__SAVE_IGS_PATH] $TMP/.opticks/igs.npy igs_path [/data1/blyth/tmp/.opticks/igs.npy] save_igs NO 
    2025-11-19 20:07:54.750 INFO  [3551660] [QSim::simulate@457]  eventID      0 igs (512, 6, 4, ) tot_ph_0 8252786688 tot_ph_0/M 8252 xxl YES MaxSlot 262000000 MaxSlot/M 262 sslice::Desc(igs_slice)
    sslice::Desc num_slice 32 TotalPhoton 8252786688 TotalPhoton/M 8252.786688
                      start    stop     offset      count    count/M 
       0 : sslice {       0,      16,         0, 257899584}257.899584
       1 : sslice {      16,      32, 257899584, 257899584}257.899584
       2 : sslice {      32,      48, 515799168, 257899584}257.899584
       3 : sslice {      48,      64, 773698752, 257899584}257.899584
       4 : sslice {      64,      80,1031598336, 257899584}257.899584
       5 : sslice {      80,      96,1289497920, 257899584}257.899584
       6 : sslice {      96,     112,1547397504, 257899584}257.899584
       7 : sslice {     112,     128,1805297088, 257899584}257.899584
       8 : sslice {     128,     144,2063196672, 257899584}257.899584
       9 : sslice {     144,     160,2321096256, 257899584}257.899584
      10 : sslice {     160,     176,2578995840, 257899584}257.899584
      11 : sslice {     176,     192,2836895424, 257899584}257.899584
      12 : sslice {     192,     208,3094795008, 257899584}257.899584
      13 : sslice {     208,     224,3352694592, 257899584}257.899584
      14 : sslice {     224,     240,3610594176, 257899584}257.899584
      15 : sslice {     240,     256,3868493760, 257899584}257.899584
      16 : sslice {     256,     272,4126393344, 257899584}257.899584
      17 : sslice {     272,     288,4384292928, 257899584}257.899584
      18 : sslice {     288,     304,4642192512, 257899584}257.899584
      19 : sslice {     304,     320,4900092096, 257899584}257.899584
      20 : sslice {     320,     336,5157991680, 257899584}257.899584
      21 : sslice {     336,     352,5415891264, 257899584}257.899584
      22 : sslice {     352,     368,5673790848, 257899584}257.899584
      23 : sslice {     368,     384,5931690432, 257899584}257.899584
      24 : sslice {     384,     400,6189590016, 257899584}257.899584
      25 : sslice {     400,     416,6447489600, 257899584}257.899584
      26 : sslice {     416,     432,6705389184, 257899584}257.899584
      27 : sslice {     432,     448,6963288768, 257899584}257.899584
      28 : sslice {     448,     464,7221188352, 257899584}257.899584
      29 : sslice {     464,     480,7479087936, 257899584}257.899584
      30 : sslice {     480,     496,7736987520, 257899584}257.899584
      31 : sslice {     496,     512,7994887104, 257899584}257.899584
                      start    stop     offset      count    count/M 
     num_slice 32
    2025-11-19 20:07:54.750 INFO  [3551660] [QSim::simulate@479]    0 : sslice {       0,      16,         0, 257899584}257.899584
    2025-11-19 20:08:32.496 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    0 dt   37.695144 slice    0 : sslice {       0,      16,         0, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51442248 merged 5511645 selected/in   0.199 merged/selected   0.107 
    2025-11-19 20:08:32.735 INFO  [3551660] [QSim::simulate@479]    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    2025-11-19 20:09:11.322 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    1 dt   38.579945 slice    1 : sslice {      16,      32, 257899584, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51430837 merged 5514117 selected/in   0.199 merged/selected   0.107 
    2025-11-19 20:09:11.549 INFO  [3551660] [QSim::simulate@479]    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    2025-11-19 20:09:50.696 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    2 dt   39.140270 slice    2 : sslice {      32,      48, 515799168, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51433012 merged 5515333 selected/in   0.199 merged/selected   0.107 
    2025-11-19 20:09:50.918 INFO  [3551660] [QSim::simulate@479]    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    2025-11-19 20:10:29.973 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    3 dt   39.049267 slice    3 : sslice {      48,      64, 773698752, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51449616 merged 5514339 selected/in   0.199 merged/selected   0.107 
    2025-11-19 20:10:30.202 INFO  [3551660] [QSim::simulate@479]    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    2025-11-19 20:11:09.239 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    4 dt   39.011671 slice    4 : sslice {      64,      80,1031598336, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51435671 merged 5512175 selected/in   0.199 merged/selected   0.107 
    2025-11-19 20:11:09.472 INFO  [3551660] [QSim::simulate@479]    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    2025-11-19 20:11:48.582 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    5 dt   39.104205 slice    5 : sslice {      80,      96,1289497920, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51441825 merged 5513278 selected/in   0.199 merged/selected   0.107 
    2025-11-19 20:11:48.812 INFO  [3551660] [QSim::simulate@479]    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    2025-11-19 20:12:27.893 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    6 dt   39.074464 slice    6 : sslice {      96,     112,1547397504, 257899584}257.899584
    ]SPM::merge_partial_select num_in 257899584 select_flagmask 8192 time_window   1.000 selected 51444895 merged 5513233 selected/in   0.199 merged/selected   0.107 
    2025-11-19 20:12:28.132 INFO  [3551660] [QSim::simulate@479]    7 : sslice {     112,     128,1805297088, 257899584}257.899584
    2025-11-19 20:13:07.205 INFO  [3551660] [QSim::simulate@505]  eventID 0 xxl YES i    7 dt   39.066645 slice    7 : sslice {     112,     128,1805297088, 257899584}257.899584

    Thread 1 "CSGOptiXSMTest" received signal SIGABRT, Aborted.
    0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libnvidia-gpucomp-580.82.07-1.el9.x86_64 libnvidia-ml-580.82.07-1.el9.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64 nvidia-driver-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff483eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4828833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff4cb135a in __cxxabiv1::__terminate (handler=<optimized out>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff4cb13c5 in std::terminate () at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff4cb1658 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff70d1b30 <typeinfo for thrust::THRUST_200302_700_890_NS::system::detail::bad_alloc>, 
        dest=0x7ffff5ecbd10 <thrust::THRUST_200302_700_890_NS::system::detail::bad_alloc::~bad_alloc()>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff5794999 in thrust::THRUST_200302_700_890_NS::cuda_cub::malloc<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/malloc_and_free.h:85
    #7  0x00007ffff57945d5 in thrust::THRUST_200302_700_890_NS::malloc<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/malloc_and_free.h:44
    #8  0x00007ffff5794036 in thrust::THRUST_200302_700_890_NS::system::detail::generic::malloc<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/detail/generic/memory.inl:59
    #9  0x00007ffff57937df in thrust::THRUST_200302_700_890_NS::malloc<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/malloc_and_free.h:56
    #10 0x00007ffff5792a07 in thrust::THRUST_200302_700_890_NS::system::detail::generic::get_temporary_buffer<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/detail/generic/temporary_buffer.inl:47
    #11 0x00007ffff5791ae0 in thrust::THRUST_200302_700_890_NS::get_temporary_buffer<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> (exec=..., n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/temporary_buffer.h:66
    #12 0x00007ffff5790f26 in thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync>::allocate (this=0x7fffffff7630, cnt=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/allocator/temporary_allocator.inl:52
    #13 0x00007ffff577fe76 in thrust::THRUST_200302_700_890_NS::detail::allocator_traits<thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::allocate(thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> >&, unsigned long)::workaround_warnings::allocate(thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> >&, unsigned long) (a=..., n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/allocator/allocator_traits.inl:378
    #14 0x00007ffff577fe9b in thrust::THRUST_200302_700_890_NS::detail::allocator_traits<thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::allocate (a=..., n=1262154623) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/allocator/allocator_traits.inl:382
    #15 0x00007ffff577ebd2 in thrust::THRUST_200302_700_890_NS::detail::contiguous_storage<unsigned char, thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::allocate (this=0x7fffffff7630, n=1262154623)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/contiguous_storage.inl:218
    #16 0x00007ffff577c557 in thrust::THRUST_200302_700_890_NS::detail::contiguous_storage<unsigned char, thrust::THRUST_200302_700_890_NS::detail::no_throw_allocator<thrust::THRUST_200302_700_890_NS::detail::temporary_allocator<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync> > >::contiguous_storage (this=0x7fffffff7630, n=1262154623, alloc=...)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/contiguous_storage.inl:76
    #17 0x00007ffff5775e3a in thrust::THRUST_200302_700_890_NS::detail::temporary_array<unsigned char, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync>::temporary_array (this=0x7fffffff7630, system=..., 
        n=1262154623) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/temporary_array.inl:84
    #18 0x00007ffff577caf2 in thrust::THRUST_200302_700_890_NS::cuda_cub::__radix_sort::radix_sort<cuda::std::__4::integral_constant<bool, true>, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, unsigned long, sphotonlite, long, thrust::THRUST_200302_700_890_NS::less<unsigned long> > (policy=..., keys=0x4c6094800, items=0x4de915600, count=51446156)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:355
    #19 0x00007ffff57764b6 in thrust::THRUST_200302_700_890_NS::cuda_cub::__smart_sort::smart_sort<cuda::std::__4::integral_constant<bool, true>, cuda::std::__4::integral_constant<bool, false>, thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite>, thrust::THRUST_200302_700_890_NS::less<unsigned long> > (
        policy=..., keys_first=..., keys_last=..., items_first=..., compare_op=...) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:461
    #20 0x00007ffff5775756 in thrust::THRUST_200302_700_890_NS::cuda_cub::sort_by_key<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite>, thrust::THRUST_200302_700_890_NS::less<unsigned long> > (policy=..., keys_first=..., keys_last=..., values=..., compare_op=...)
        at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:551
    #21 0x00007ffff5775016 in thrust::THRUST_200302_700_890_NS::cuda_cub::sort_by_key<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite> > (policy=..., keys_first=..., keys_last=..., values=...) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/system/cuda/detail/sort.h:621
    #22 0x00007ffff5773e0e in thrust::THRUST_200302_700_890_NS::sort_by_key<thrust::THRUST_200302_700_890_NS::cuda_cub::execute_on_stream_nosync, thrust::THRUST_200302_700_890_NS::device_ptr<unsigned long>, thrust::THRUST_200302_700_890_NS::device_ptr<sphotonlite> > (exec=..., keys_first=..., keys_last=..., values_first=...) at /usr/local/cuda-12.4/targets/x86_64-linux/include/thrust/detail/sort.inl:102
    #23 0x00007ffff576cc7f in SPM::merge_partial_select (d_in=0x7ff9fa000000, num_in=257899584, d_out=0x166c7a90, num_out=0x166c7a08, select_flagmask=8192, time_window=1, stream=0x0) at /home/blyth/opticks/sysrap/SPM.cu:266
    #24 0x00007ffff5ec0e61 in QEvt::PerLaunchMerge (evt=0x166c7960, stream=0x0) at /home/blyth/opticks/qudarap/QEvt.cc:1112
    #25 0x00007ffff5ec03fd in QEvt::gatherHitLiteMerged (this=0x186569c0) at /home/blyth/opticks/qudarap/QEvt.cc:1033
    #26 0x00007ffff5ec17a1 in QEvt::gatherComponent_ (this=0x186569c0, cmp=67108864) at /home/blyth/opticks/qudarap/QEvt.cc:1295
    --Type <RET> for more, q to quit, c to continue without paging--



SPM.cu::

    261     thrust::copy_n(policy, selected, num_selected          , thrust::device_ptr<sphotonlite>(d_vals));
    262 
    263 
    264     // 4. sort_by_key arranging hits with same (id, timebucket) to be contiguous
    265 
    266     thrust::sort_by_key(policy,
    267                         thrust::device_ptr<uint64_t>(d_keys),
    268                         thrust::device_ptr<uint64_t>(d_keys + num_selected),
    269                         thrust::device_ptr<sphotonlite>(d_vals));
    270 






