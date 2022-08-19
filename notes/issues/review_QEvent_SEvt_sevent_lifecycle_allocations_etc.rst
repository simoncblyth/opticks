review_QEvent_SEvt_sevent_lifecycle_allocations_etc
=======================================================


TODO : try ntds3 with record recording and saving
------------------------------------------------------



TODO : ntds3 run with TMP=$HOME/.opticks for more permanent geometry base for gxs.sh gxt.sh gxr.sh tests
-----------------------------------------------------------------------------------------------------------

::

    export -n QEvent

    export TMP=$HOME/.opticks SEvt=INFO SEventConfig=INFO CSG_GGeo_Convert=INFO
    ntds3


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

