review_QEvent_SEvt_sevent_lifecycle_allocations_etc
=======================================================


TODO : try ntds3 with record recording and saving
------------------------------------------------------



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

