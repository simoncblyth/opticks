review_QEvent_SEvt_sevent_lifecycle_allocations_etc
=======================================================


TODO : try ntds3 with record recording and saving
------------------------------------------------------



TODO : make the SVN commits
-------------------------------



TODO : try opticksMode 1 
--------------------------

::

    2022-08-20 03:02:39.567 INFO  [229251] [junoSD_PMT_v2_Opticks::EndOfEvent@258] ] num_hit 3825 merged_count  0 savehit_count  3825 m_merged_total 0 m_savehit_total 3793 m_opticksMode 3 LEVEL 5:DEBUG
    2022-08-20 03:02:39.567 INFO  [229251] [junoSD_PMT_v2_Opticks::TerminateEvent@307]  invoking SEvt::Clear as no U4Recorder detected 
    2022-08-20 03:02:39.567 INFO  [229251] [SEvt::clear@411] [
    2022-08-20 03:02:39.567 INFO  [229251] [SEvt::clear@428] ]
    ] junoSD_PMT_v2::EndOfEvent m_opticksMode  3
    junoSD_PMT_v2::EndOfEvent m_opticksMode 3 hitCollection 5335 hitCollection_muon 0 hitCollection_opticks 0
    junotoptask:DetSimAlg.finalize  INFO: DetSimAlg finalized successfully
    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::finalizeOpticks m_opticksMode 3 WITH_G4CXOPTICKS 
    junotoptask:PMTSimParamSvc.finalize  INFO: PMTSimParamSvc is finalizing!



DONE : gxt.sh grablog analog iteration : plucking low hanging fruit : full JUNO total runtime now under 1s 
-------------------------------------------------------------------------------------------------------------------

::

    epsilon:~ blyth$ gx
    /Users/blyth/opticks/g4cx
    epsilon:g4cx blyth$ ./gxt.sh grablog 
    epsilon:g4cx blyth$ ./gxt.sh analog

Log lines with delta time more than 2% of total time::

    In [2]: log[2]
    Out[2]: 
                     timestamp :        DTS-prev :        DFS-frst :msg
    2022-08-23 22:20:40.570000 :      0.3100[31] :      0.3320[34] : INFO  [53026] [QBase::init@53] ] QU::UploadArray 
    2022-08-23 22:20:40.749000 :      0.1790[18] :      0.5110[52] : INFO  [53026] [QSim::UploadComponents@111] ] new QRng 
    2022-08-23 22:20:40.878000 :      0.1010[10] :      0.6400[65] : INFO  [53026] [CSGOptiX::initCtx@322] ]
    2022-08-23 22:20:40.912000 :      0.0340[ 3] :      0.6740[68] : INFO  [53026] [CSGOptiX::initPIP@333] ]
    2022-08-23 22:20:41.015000 :      0.0350[ 4] :      0.7770[79] : INFO  [53026] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-23 22:20:41.189000 :      0.1140[12] :      0.9510[97] : INFO  [53026] [SEvt::gather@1378]  k        simtrace a  <f4(627000, 4, 4, )
    2022-08-23 22:20:41.222000 :      0.0320[ 3] :      0.9840[100] : INFO  [53026] [SEvt::save@1505] ] fold.save 


                             - :                 :                 :G4CXSimtraceTest.log
    2022-08-23 22:20:40.238000 :                 :                 :start
    2022-08-23 22:20:41.223000 :                 :                 :end
                             - :                 :      0.9850[100] :total seconds
                             - :                 :      2.0000[100] :pc_cut


Using CUDA_VISIBLE_DEVICES=0 or 1 reduces that a little::

    In [1]: log[2]
    Out[1]: 
                     timestamp :        DTS-prev :        DFS-frst :msg
    2022-08-23 22:30:41.646000 :      0.2690[29] :      0.2970[32] : INFO  [53509] [QBase::init@53] ] QU::UploadArray 
    2022-08-23 22:30:41.834000 :      0.1880[20] :      0.4850[53] : INFO  [53509] [QSim::UploadComponents@111] ] new QRng 
    2022-08-23 22:30:41.938000 :      0.0780[ 8] :      0.5890[64] : INFO  [53509] [CSGOptiX::initCtx@322] ]
    2022-08-23 22:30:41.964000 :      0.0260[ 3] :      0.6150[67] : INFO  [53509] [CSGOptiX::initPIP@333] ]
    2022-08-23 22:30:42.063000 :      0.0340[ 4] :      0.7140[78] : INFO  [53509] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-23 22:30:42.237000 :      0.1150[12] :      0.8880[96] : INFO  [53509] [SEvt::gather@1378]  k        simtrace a  <f4(627000, 4, 4, )
    2022-08-23 22:30:42.269000 :      0.0320[ 3] :      0.9200[100] : INFO  [53509] [SEvt::save@1505] ] fold.save 


                             - :                 :                 :G4CXSimtraceTest.log
    2022-08-23 22:30:41.349000 :                 :                 :start
    2022-08-23 22:30:42.270000 :                 :                 :end
                             - :                 :      0.9210[100] :total seconds
                             - :                 :      2.0000[100] :pc_cut




TODO : gxs.sh gxt.sh gxr.sh running from saved geometry with relocated CSGFoundry/SSim/stree
------------------------------------------------------------------------------------------------

::

   export SEventConfig=INFO QU=INFO



::

    2022-08-21 04:12:43.408 INFO  [303833] [QU::device_alloc@194]  num_items    3000000 size   12000000 label device_alloc_genstep:int seed
    2022-08-21 04:12:43.408 INFO  [303833] [SEvt::setNumPhoton@576]  num_photon 1000
    2022-08-21 04:12:43.409 INFO  [303833] [SEvt::setNumPhoton@589]  evt->num_photon 1000 evt->num_tag 1000 evt->num_flat 1000
    2022-08-21 04:12:43.409 INFO  [303833] [QU::device_alloc_zero@229]  num_items    3000000 size  192000000 label      max_photon
    2022-08-21 04:12:43.409 INFO  [303833] [QU::device_alloc_zero@229]  num_items   30000000 size 1920000000 label max_photon*max_record
    2022-08-21 04:12:43.411 INFO  [303833] [QU::device_alloc_zero@229]  num_items   30000000 size  480000000 label max_photon*max_rec
    2022-08-21 04:12:43.412 INFO  [303833] [QU::device_alloc_zero@229]  num_items   30000000 size  480000000 label max_photon*max_seq
    2022-08-21 04:12:43.413 INFO  [303833] [QU::device_alloc_zero@229]  num_items   30000000 size  960000000 label max_photon*max_prd
    2022-08-21 04:12:43.414 INFO  [303833] [QU::device_alloc_zero@229]  num_items   72000000 size 2304000000 label max_photon*max_tag
    2022-08-21 04:12:43.417 INFO  [303833] [QU::device_alloc_zero@229]  num_items   72000000 size 18432000000 label max_photon*max_flat
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (cudaMalloc(reinterpret_cast<void**>( &d ), size ) ) failed with error: 'out of memory' (/data/blyth/junotop/opticks/qudarap/QU.cc:236)



Looks like flat allocation has lost touch with actual usage ? 

Suspect the switch to compound stag.h/sflat not reflected by the allocation. 

::

    epsilon:tests blyth$ ./SEventConfigTest.sh 
    2022-08-21 17:16:37.252 INFO  [34597489] [test_EstimateAlloc@20] 
    SEventConfig::Desc
            OPTICKS_EVENTMODE          EventMode  : StandardFullDebug
          OPTICKS_MAX_GENSTEP         MaxGenstep  : 1000000
           OPTICKS_MAX_PHOTON          MaxPhoton  : 1000000
         OPTICKS_MAX_SIMTRACE        MaxSimtrace  : 1000000
           OPTICKS_MAX_BOUNCE          MaxBounce  : 9
           OPTICKS_MAX_RECORD          MaxRecord  : 10
              OPTICKS_MAX_REC             MaxRec  : 10
              OPTICKS_MAX_SEQ             MaxSeq  : 10
              OPTICKS_MAX_PRD             MaxPrd  : 10
              OPTICKS_MAX_TAG             MaxTag  : 24
             OPTICKS_MAX_FLAT            MaxFlat  : 24
             OPTICKS_HIT_MASK            HitMask  : 64
                                    HitMaskLabel  : SD
           OPTICKS_MAX_EXTENT          MaxExtent  : 1000
             OPTICKS_MAX_TIME            MaxTime  : 10
              OPTICKS_RG_MODE             RGMode  : 2
                                     RGModeLabel  : simulate
            OPTICKS_COMP_MASK           CompMask  : 12670
                                   CompMaskLabel  : genstep,photon,record,rec,seq,prd,hit,tag,flat
             OPTICKS_OUT_FOLD            OutFold  : $DefaultOutputDir
             OPTICKS_OUT_NAME            OutName  : -
    OPTICKS_PROPAGATE_EPSILON   PropagateEpsilon  :     0.0500
         OPTICKS_INPUT_PHOTON        InputPhoton  : -

    al.desc
    salloc::desc alloc.size 11 label.size 11

         [           size   num_items sizeof_item       spare]    size_GB    percent label

         [            208           1         208           0]       0.00       0.00 QEvent::QEvent/sevent
         [        8294400     2073600           4           0]       0.01       0.03 Frame::DeviceAllo:num_pixels
         [       96000000     1000000          96           0]       0.10       0.39 device_alloc_genstep:quad6
         [       12000000     3000000           4           0]       0.01       0.05 device_alloc_genstep:int seed
         [      192000000     3000000          64           0]       0.19       0.77 max_photon
         [     1920000000    30000000          64           0]       1.92       7.72 max_photon*max_record
         [      480000000    30000000          16           0]       0.48       1.93 max_photon*max_rec
         [      480000000    30000000          16           0]       0.48       1.93 max_photon*max_seq
         [      960000000    30000000          32           0]       0.96       3.86 max_photon*max_prd
         [     2304000000    72000000          32           0]       2.30       9.26 max_photon*max_tag
         [    18432000000    72000000         256           0]      18.43      74.07 max_photon*max_flat

     tot      24884294608                                           24.88
    epsilon:tests blyth$ 




    epsilon:tests blyth$ ./salloc_test.sh build_run
    a.desc
    salloc::desc alloc.size 11 label.size 11

         [           size   num_items sizeof_item       spare]    size_GB    percent label

         [            208           1         208           0]       0.00       0.00 QEvent::QEvent/sevent
         [        8294400     2073600           4           0]       0.01       0.03 Frame::DeviceAllo:num_pixels
         [       96000000     1000000          96           0]       0.10       0.39 device_alloc_genstep:quad6
         [       12000000     3000000           4           0]       0.01       0.05 device_alloc_genstep:int seed
         [      192000000     3000000          64           0]       0.19       0.77 max_photon
         [     1920000000    30000000          64           0]       1.92       7.72 max_photon*max_record
         [      480000000    30000000          16           0]       0.48       1.93 max_photon*max_rec
         [      480000000    30000000          16           0]       0.48       1.93 max_photon*max_seq
         [      960000000    30000000          32           0]       0.96       3.86 max_photon*max_prd
         [     2304000000    72000000          32           0]       2.30       9.26 max_photon*max_tag
         [    18432000000    72000000         256           0]      18.43      74.07 max_photon*max_flat

     tot      24884294608                                           24.88
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 



::

    429 NP* QEvent::gatherFlat() const
    430 {
    431     if(!hasFlat()) LOG(LEVEL) << " gatherFlat called when there is no such array, use SEventConfig::SetCompMask to avoid " ;
    432     if(!hasFlat()) return nullptr ;
    433 
    434     NP* flat = sev->makeFlat();
    435     LOG(LEVEL) << " evt.num_flat " << evt->num_flat << " flat.desc " << flat->desc() ;
    436     QU::copy_device_to_host<sflat>( (sflat*)flat->bytes(), evt->flat, evt->num_flat );
    437     return flat ;
    438 }

    1198 NP* SEvt::makeFlat() const
    1199 {
    1200     return NP::Make<float>( evt->num_photon, sflat::SLOTS );   // 
    1201 }


    202 struct sflat
    203 {
    204     static constexpr const unsigned SLOTS = stag::SLOTS ;
    205     float flat[SLOTS] ;    // sizeof(sflat) = 4*64 = 256 bytes with SLOTS = 64 
    206 };
    207 




CSGFoundry::inst_find_unique taking lots of time (21s) for little benefit
------------------------------------------------------------------------------

Avoid finding unique ins_index, sensor_identifier, sensor_index as those
are not used.  Shaving 21s::

    2022-08-22 03:03:32.535 INFO  [378584] [CSGFoundry::upload@2615] [ inst_find_unique 
    2022-08-22 03:03:32.539 INFO  [378584] [CSGFoundry::upload@2617] ] inst_find_unique 


::

    2022-08-22 02:29:24.822 INFO  [364740] [CSGOptiX::InitGeo@168] [
    2022-08-22 02:29:24.822 INFO  [364740] [CSGFoundry::upload@2610] [ inst_find_unique 
    2022-08-22 02:29:45.208 INFO  [364740] [CSGFoundry::upload@2612] ] inst_find_unique 
    2022-08-22 02:29:45.209 INFO  [364740] [CSGFoundry::upload@2613] CSGFoundry  num_total 10 num_solid 10 num_prim 3248 num_node 23518 num_plan 0 num_tran 8159 num_itra 8159 num_inst 48477 ins 48477 gas 10 sensor_identifier 45613 sensor_index 45613 meshname 139 mmlabel 10 mtime 1661012280 mtimestamp 20220821_001800 sim Y
    2022-08-22 02:29:45.209 INFO  [364740] [CSGFoundry::upload@2622] [ CU::UploadArray 
    2022-08-22 02:29:45.219 INFO  [364740] [CSGFoundry::upload@2627] ] CU::UploadArray 
    2022-08-22 02:29:45.219 INFO  [364740] [CSGFoundry::upload@2638] ]
    2022-08-22 02:29:45.219 INFO  [364740] [CSGOptiX::InitGeo@170] ]



DONE : ntds3 run with TMP=$HOME/.opticks for more permanent geometry base saved geom running test
-----------------------------------------------------------------------------------------------------------

::

    export -n QEvent

    export TMP=$HOME/.opticks SEvt=INFO SEventConfig=INFO CSG_GGeo_Convert=INFO
    ntds3

    TMP=$HOME/.opticks ntds3   ## redo: following stree relocation to CSGFoundry/SSim/stree 


::

    N[blyth@localhost opticks]$ l /home/blyth/.opticks/ntds3/G4CXOpticks/
    total 41016
        0 drwxr-xr-x.  5 blyth blyth      122 Aug 19 20:22 .
    20504 -rw-rw-r--.  1 blyth blyth 20992919 Aug 19 20:22 origin.gdml
        4 -rw-rw-r--.  1 blyth blyth      198 Aug 19 20:22 origin_gdxml_report.txt
    20504 -rw-rw-r--.  1 blyth blyth 20994471 Aug 19 20:22 origin_raw.gdml
        0 drwxrwxr-x. 15 blyth blyth      273 Aug 19 20:22 GGeo
        0 drwxr-xr-x.  3 blyth blyth      190 Aug 19 20:22 CSGFoundry
        4 drwxr-xr-x.  4 blyth blyth     4096 Aug 19 20:22 stree
        0 drwxr-xr-x.  3 blyth blyth       25 Aug 19 20:22 ..
    N[blyth@localhost opticks]$ 





TODO : check merged counts
----------------------------

::

    2022-08-19 19:49:11.243 INFO  [175864] [QEvent::gatherComponent@563] [ comp 2
    2022-08-19 19:49:11.244 INFO  [175864] [QEvent::gatherComponent@567] [ comp 2 proceed 1 a 0x7fff36f3e750
    2022-08-19 19:49:11.244 INFO  [175864] [SEvt::gather@1373]  k         genstep a  <f4(116, 6, 4, )
    2022-08-19 19:49:11.244 INFO  [175864] [QEvent::gatherComponent@563] [ comp 4
    2022-08-19 19:49:11.247 INFO  [175864] [QEvent::gatherPhoton@355] [ evt.num_photon 10956 p.sstr (10956, 4, 4, ) evt.photon 0x7fff2a000000
    2022-08-19 19:49:11.247 INFO  [175864] [QEvent::gatherPhoton@358] ] evt.num_photon 10956
    2022-08-19 19:49:11.247 INFO  [175864] [QEvent::gatherComponent@567] [ comp 4 proceed 1 a 0x7fff36d78a30
    2022-08-19 19:49:11.247 INFO  [175864] [SEvt::gather@1373]  k          photon a  <f4(10956, 4, 4, )
    2022-08-19 19:49:11.247 INFO  [175864] [QEvent::gatherComponent@563] [ comp 256
    2022-08-19 19:49:11.255 INFO  [175864] [QEvent::gatherHit@523]  evt.photon 0x7fff2a000000 evt.num_photon 10956 evt.num_hit 3793 selector.hitmask 64 SEventConfig::HitMask 64 SEventConfig::HitMaskLabel SD
    2022-08-19 19:49:11.256 INFO  [175864] [QEvent::gatherHit_@550]  hit.sstr (3793, 4, 4, )
    2022-08-19 19:49:11.256 INFO  [175864] [QEvent::gatherComponent@567] [ comp 256 proceed 1 a 0x7fff36f14680
    2022-08-19 19:49:11.256 INFO  [175864] [SEvt::gather@1373]  k             hit a  <f4(3793, 4, 4, )
    2022-08-19 19:49:11.256 INFO  [175864] [junoSD_PMT_v2_Opticks::EndOfEvent@187]  eventID 0 num_hit 3793 way_enabled 0
         0 gp.x   17412.67 gp.y    2617.35 gp.z    8039.29 gp.R   19356.70 pmt    4938             SC|SD|BT
         1 gp.x     786.81 gp.y   19319.34 gp.z      22.82 gp.R   19335.37 pmt    8753          RE|SC|SD|BT
         2 gp.x  -19149.59 gp.y    -225.21 gp.z   -2113.11 gp.R   19267.14 pmt    9656             RE|SD|BT
         3 gp.x   -3375.24 gp.y  -18937.39 gp.z   -1623.00 gp.R   19304.17 pmt    9485          RE|SC|SD|BT
         4 gp.x  -13073.55 gp.y    7722.22 gp.z  -11968.27 gp.R   19333.65 pmt   14395             SC|SD|BT
         5 gp.x  -17010.48 gp.y    9089.52 gp.z    1732.15 gp.R   19364.30 pmt  323327             SC|SD|BT
         6 gp.x   -1600.75 gp.y   18540.89 gp.z    5044.77 gp.R   19281.51 pmt    6397                SD|BT
         7 gp.x   16901.99 gp.y    5818.85 gp.z   -7339.26 gp.R   19323.59 pmt   12289                SD|BT
         8 gp.x  -15630.24 gp.y   -8382.87 gp.z    7747.10 gp.R   19354.44 pmt    5037                SD|BT
         9 gp.x   14382.95 gp.y  -11262.46 gp.z    6144.60 gp.R   19273.51 pmt    5922             RE|SD|BT
        10 gp.x  -14453.04 gp.y    6286.11 gp.z   11080.78 gp.R   19266.28 pmt    3709             RE|SD|BT
        11 gp.x   -8930.25 gp.y    2609.81 gp.z   16956.24 gp.R   19341.00 pmt    1006             SC|SD|BT
        12 gp.x   15875.78 gp.y   10492.09 gp.z    2980.17 gp.R   19261.51 pmt    7436             RE|SD|BT
        13 gp.x   -6810.50 gp.y   18017.92 gp.z     159.78 gp.R   19262.76 pmt    8765          RE|SC|SD|BT
        14 gp.x  -10050.34 gp.y   13946.82 gp.z   -8951.06 gp.R   19381.55 pmt   13127                SD|BT
        15 gp.x    -421.58 gp.y  -18617.98 gp.z   -5014.34 gp.R   19286.02 pmt   11212                SD|BT
        16 gp.x    5027.51 gp.y   16938.40 gp.z    7942.81 gp.R   19371.97 pmt    5166                SD|BT
        17 gp.x   16452.79 gp.y   10116.41 gp.z    -165.36 gp.R   19314.85 pmt    8918             RE|SD|BT
        18 gp.x   -1135.70 gp.y  -17112.09 gp.z   -8945.52 gp.R   19342.59 pmt   13011                SD|BT
        19 gp.x   15908.88 gp.y    9048.97 gp.z    6199.24 gp.R   19323.74 pmt    5961             SC|SD|BT
    2022-08-19 19:49:11.311 INFO  [175864] [junoSD_PMT_v2_Opticks::EndOfEvent@255] ] num_hit 3793 merged_count  0 m_merged_total 0 m_opticksMode 3

    ...
        16 gp.x   15788.26 gp.y    8861.23 gp.z    6633.64 gp.R   19282.01 pmt    5756             SC|SD|BT
        17 gp.x   -4783.42 gp.y   -1058.68 gp.z  -18648.80 gp.R   19281.59 pmt   17312                SD|BT
        18 gp.x  -13847.39 gp.y   -6365.28 gp.z  -11791.05 gp.R   19269.04 pmt   14248          RE|SC|SD|BT
        19 gp.x   14849.45 gp.y   -2178.64 gp.z   12074.18 gp.R   19262.35 pmt    3108                SD|BT
    2022-08-19 19:49:12.309 INFO  [175864] [junoSD_PMT_v2_Opticks::EndOfEvent@255] ] num_hit 3825 merged_count  0 m_merged_total 0 m_opticksMode 3
    2022-08-19 19:49:12.309 INFO  [175864] [junoSD_PMT_v2_Opticks::TerminateEvent@300]  invoking SEvt::Clear as no U4Recorder detected 
    2022-08-19 19:49:12.309 INFO  [175864] [SEvt::clear@411] [
    2022-08-19 19:49:12.310 INFO  [175864] [SEvt::clear@428] ]
    ] junoSD_PMT_v2::EndOfEvent m_opticksMode  3
    junoSD_PMT_v2::EndOfEvent m_opticksMode 3 hitCollection 5335 hitCollection_muon 0 hitCollection_opticks 0
    junotoptask:DetSimAlg.finalize  INFO: DetSimAlg finalized successfully
    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::finalizeOpticks m_opticksMode 3 WITH_G4CXOPTICKS 
    junotoptask:PMTSimParamSvc.finalize  INFO: PMTSimParamSvc is finalizing!
    junotoptask.finalize            INFO: events processed 2
    Delete G4SvcRunManager
    2022-08-19 19:49:12.553 DEBUG [175864] [junoSD_PMT_v2_Opticks::~junoSD_PMT_v2_Opticks@71]  WITH_G4CXOPTICKS  m_opticksMode 3 m_event_total 2 m_genstep_total 218 m_photon_total 21909 m_hit_total 7618 m_merged_total 0




::

    2022-08-19 20:24:49.418 INFO  [178849] [SEvt::clear@428] ]
    ] junoSD_PMT_v2::EndOfEvent m_opticksMode  3
    junoSD_PMT_v2::EndOfEvent m_opticksMode 3 hitCollection 5335 hitCollection_muon 0 hitCollection_opticks 0
    junotoptask:DetSimAlg.finalize  INFO: DetSimAlg finalized successfully
    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::finalizeOpticks m_opticksMode 3 WITH_G4CXOPTICKS 
    junotoptask:PMTSimParamSvc.finalize  INFO: PMTSimParamSvc is finalizing!
    junotoptask.finalize            INFO: events processed 2




Note that the alloc to max was formerly only done for photon
---------------------------------------------------------------

::

    632 void QEvent::device_alloc_photon()
    633 {   
    634     evt->photon  = evt->max_photon > 0 ? QU::device_alloc_zero<sphoton>( evt->max_photon ) : nullptr ;
    635     
    636     evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_photon * evt->max_record ) : nullptr ;
    637     evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_photon * evt->max_rec    ) : nullptr ;
    638     evt->seq     = evt->max_seq    > 0 ? QU::device_alloc_zero<sseq>(    evt->max_photon * evt->max_seq    ) : nullptr ;
    639     evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_photon * evt->max_prd    ) : nullptr ;
    640     evt->tag     = evt->max_tag    > 0 ? QU::device_alloc_zero<stag>(    evt->max_photon * evt->max_tag    ) : nullptr ;
    641     evt->flat    = evt->max_flat   > 0 ? QU::device_alloc_zero<sflat>(   evt->max_photon * evt->max_flat   ) : nullptr ;
    642     
    643     /*
    644     evt->record  = evt->num_record > 0 ? QU::device_alloc_zero<sphoton>( evt->num_record ) : nullptr ; 
    645     evt->rec     = evt->num_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->num_rec  )   : nullptr ; 
    646     evt->seq     = evt->num_seq    > 0 ? QU::device_alloc_zero<sseq>(    evt->num_seq  )   : nullptr ; 
    647     evt->prd     = evt->num_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->num_prd  )   : nullptr ; 
    648     evt->tag     = evt->num_tag    > 0 ? QU::device_alloc_zero<stag>(    evt->num_tag  )   : nullptr ; 
    649     evt->flat    = evt->num_flat   > 0 ? QU::device_alloc_zero<sflat>(   evt->num_flat  )  : nullptr ; 
    650     */
    651 



TODO: logging rationalize QEvent=INFO SEvt=INFO
-------------------------------------------------

Where to call the below in integrated running::

   SEventConfig::SetCompMask("photon,genstep,hit"); 


Need coordination/consistency between the max and the comps


::


    2022-08-18 19:07:24.113 INFO  [91491] [SEvt::gather@1372]  comp 2 k genstep comp_skip 0
    2022-08-18 19:07:24.113 INFO  [91491] [QEvent::gatherComponent@563] [ comp 2
    2022-08-18 19:07:24.113 INFO  [91491] [QEvent::gatherComponent@567] [ comp 2 proceed 1 a 0x7fff366647b0
    2022-08-18 19:07:24.113 INFO  [91491] [SEvt::gather@1375]  a  <f4(102, 6, 4, )
    2022-08-18 19:07:24.113 INFO  [91491] [SEvt::gather@1372]  comp 4 k photon comp_skip 0
    2022-08-18 19:07:24.113 INFO  [91491] [QEvent::gatherComponent@563] [ comp 4
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherPhoton@355] [ evt.num_photon 10953 p.sstr (10953, 4, 4, ) evt.photon 0x7fff2a000000
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherPhoton@358] ] evt.num_photon 10953
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@567] [ comp 4 proceed 1 a 0x7fff3668dfb0
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1375]  a  <f4(10953, 4, 4, )
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1372]  comp 8 k record comp_skip 0
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@563] [ comp 8
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherRecord@443]  gatherRecord called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@567] [ comp 8 proceed 1 a 0
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1375]  a -
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1372]  comp 16 k rec comp_skip 0
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@563] [ comp 16
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherRec@455]  gatherRec called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@567] [ comp 16 proceed 1 a 0
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1375]  a -
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1372]  comp 32 k seq comp_skip 0
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@563] [ comp 32
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherSeq@398]  gatherSeq called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@567] [ comp 32 proceed 1 a 0
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1375]  a -
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1372]  comp 64 k prd comp_skip 0
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@563] [ comp 64
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherPrd@409]  gatherPrd called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@567] [ comp 64 proceed 1 a 0
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1375]  a -
    2022-08-18 19:07:24.115 INFO  [91491] [SEvt::gather@1372]  comp 128 k seed comp_skip 0
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@563] [ comp 128
    2022-08-18 19:07:24.115 INFO  [91491] [QEvent::gatherComponent@567] [ comp 128 proceed 1 a 0x7fff366928d0
    2022-08-18 19:07:24.116 INFO  [91491] [SEvt::gather@1375]  a  <i4(10953, )
    2022-08-18 19:07:24.116 INFO  [91491] [SEvt::gather@1372]  comp 256 k hit comp_skip 0
    2022-08-18 19:07:24.116 INFO  [91491] [QEvent::gatherComponent@563] [ comp 256

