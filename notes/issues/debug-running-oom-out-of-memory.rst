debug-running-oom-out-of-memory
===================================


HMM: now that using detected VRAM to auto-configure max_slot it makes 
no sense for debug records to use the same max_slot

jok-tds-gdb::

    2025-01-08 16:15:25.026 INFO  [8527] [U4Recorder::PreUserTrackingAction_Optical@442]  modulo 100000 : ulabel.id 0
    2025-01-08 16:15:25.116 INFO  [8527] [QSim::simulate@395] sslice {    0,    1,      0,  10000}
    2025-01-08 16:15:25.155 ERROR [8527] [QU::_cudaMalloc@260] save salloc record to /data/blyth/opticks/GEOM/J_2025jan08/jok-tds
    junotoptask.execute            ERROR: CUDA call (max_slot*max_record*sizeof(sphoton) ) failed with error: 'out of memory' (/home/blyth/opticks/qudarap/QU.cc:253)
    [salloc::desc alloc.size 29 label.size 29
    [salloc.meta
    evt.max_curand:1000000000
    evt.max_slot:197000000
    evt.max_photon:1000000
    evt.num_photon:10000
    evt.max_curand/M:1000
    evt.max_slot/M:197
    evt.max_photon/M:1
    evt.num_photon/M:0
    evt.max_record:32
    evt.max_rec:0
    evt.max_seq:1
    evt.max_prd:0
    evt.max_tag:0
    evt.max_flat:0
    evt.num_record:320000
    evt.num_rec:0
    evt.num_seq:10000
    evt.num_prd:0
    evt.num_tag:0
    evt.num_flat:0
    ]salloc.meta

         [           size   num_items sizeof_item       spare]    size_GB    percent label
         [        (bytes)                                    ]   size/1e9            

         [              8           1           8           0]       0.00       0.00 QBase::init/d_base
         [             24           1          24           0]       0.00       0.00 QRng::initMeta/d_qr
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             32           1          32           0]       0.00       0.00 QBnd::QBnd/d_qb
         [            432           1         432           0]       0.00       0.00 QDebug::QDebug/d_dbg
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             24           1          24           0]       0.00       0.00 QScint::QScint/d_scint
         [             24           1          24           0]       0.00       0.00 QCerenkov::QCerenkov/d_cerenkov.0
         [           2880         720           4           0]       0.00       0.00 QProp::upload/pp
         [             16           1          16           0]       0.00       0.00 QProp::upload/d_prop
         [           1056         264           4           0]       0.00       0.00 QProp::upload/pp
         [             16           1          16           0]       0.00       0.00 QProp::upload/d_prop
         [             48          12           4           0]       0.00       0.00 QPMT::init_thickness/d_thickness
         [         140896       35224           4           0]       0.00       0.00 QPMT::init_lcqs/d_lcqs
         [             40           1          40           0]       0.00       0.00 QPMT::init/d_pmt
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             96           1          96           0]       0.00       0.00 QMultiFilm::uploadMultifilmlut
         [            256           1         256           0]       0.00       0.00 QEvent::QEvent/sevent
         [             64           1          64           0]       0.00       0.00 QSim::init.sim
         [        8294400     2073600           4           0]       0.01       0.01 Frame::DeviceAllo:num_pixels
         [      960000000    10000000          96           0]       0.96       0.68 QEvent::setGenstep/device_alloc_genstep_and_seed:quad6
         [        4000000     1000000           4           0]       0.00       0.00 QEvent::setGenstep/device_alloc_genstep_and_seed:int seed
         [    12608000000   197000000          64           0]      12.61       8.87 QEvent::device_alloc_photon/max_slot*sizeof(sphoton)
         [   128578093056  2009032704          64           0]     128.58      90.45 max_slot*max_record*sizeof(sphoton)

     tot     142158533880                                          142.16
    ]salloc::desc

    junotoptask.finalize            WARN: invalid state tranform ((Running)) => ((Finalized))
    [2025-01-08 16:15:25,158] p8527 {/data/blyth/junotop/junosw/InstallArea/python/Tutorial/JUNOApplication.py:201} INFO - ]JUNOApplication.run
    [Thread 0x7fffd03dd700 (LWP 8729) exited]
    junotoptask.finalize            WARN: invalid state tranform ((Running)) => ((Finalized))
    junotoptask.terminate           WARN: terminate ignored due to errors



No stack because of embedded running by should be from first QEvent::setNumPhoton with debug arrays enabled::

     389     for(int i=0 ; i < num_slice ; i++)
     390     {
     391         SProf::Add("QSim__simulate_PRUP");
     392 
     393         const sslice& sl = igs_slice[i] ;
     394 
     395         LOG(info) << sl.desc() ;
     396 
     397         int rc = event->setGenstepUpload_NP(igs, &sl ) ;



::

     946 void QEvent::setNumPhoton(unsigned num_photon )
     947 {
     948     LOG_IF(info, LIFECYCLE) << " num_photon " << num_photon ;
     949     LOG(LEVEL);
     950 
     951     sev->setNumPhoton(num_photon);
     952     if( evt->photon == nullptr ) device_alloc_photon();
     953     uploadEvt();
     954 }



::

    0978 void QEvent::device_alloc_photon()
     979 {
     980     LOG_IF(info, LIFECYCLE) ;
     981     SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
     982 
     983     LOG(LEVEL)
     984         << " evt.max_slot   " << evt->max_slot
     985         << " evt.max_photon " << evt->max_photon
     986         << " evt.num_photon " << evt->num_photon
     987 #ifndef PRODUCTION
     988         << " evt.num_record " << evt->num_record
     989         << " evt.num_rec    " << evt->num_rec
     990         << " evt.num_seq    " << evt->num_seq
     991         << " evt.num_prd    " << evt->num_prd
     992         << " evt.num_tag    " << evt->num_tag
     993         << " evt.num_flat   " << evt->num_flat
     994 #endif
     995         ;
     996 
     997     evt->photon  = evt->max_slot > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot, "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)" ) : nullptr ;
     998 
     999 #ifndef PRODUCTION
    1000     evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot * evt->max_record, "max_slot*max_record*sizeof(sphoton)" ) : nullptr ;
    1001     evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_slot * evt->max_rec   , "max_slot*max_rec*sizeof(srec)"    ) : nullptr ;
    1002     evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_slot * evt->max_prd   , "max_slot*max_prd*sizeof(quad2)"    ) : nullptr ;
    1003     evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_slot                  , "max_slot*sizeof(sseq)"    ) : nullptr ;
    1004     evt->tag     = evt->max_tag   == 1 ? QU::device_alloc_zero<stag>(    evt->max_slot                  , "max_slot*sizeof(stag)"    ) : nullptr ;
    1005     evt->flat    = evt->max_flat  == 1 ? QU::device_alloc_zero<sflat>(   evt->max_slot                  , "max_slot*sizeof(sflat)"   ) : nullptr ;
    1006 #endif
    1007 
    1008     LOG(LEVEL) << desc() ;
    1009     LOG(LEVEL) << desc_alloc() ;
    1010 }




Hmm could complicate this with different max for debug arrays ? Or just say that debug running 
needs to set something reasonable, not leave to default of zero which will try to fill VRAM::

    export OPTICKS_MAX_SLOT=M1


::

    P[blyth@localhost opticks]$ git diff qudarap/QU.cc
    diff --git a/qudarap/QU.cc b/qudarap/QU.cc
    index 06781ab6c..2a3e6109a 100644
    --- a/qudarap/QU.cc
    +++ b/qudarap/QU.cc
    @@ -239,8 +239,20 @@ template QUDARAP_API void  QU::device_free_and_alloc<uchar4>(uchar4** dd, unsign
     template QUDARAP_API void  QU::device_free_and_alloc<float4>(float4** dd, unsigned num_items) ;
     template QUDARAP_API void  QU::device_free_and_alloc<quad4>(quad4** dd, unsigned num_items) ;
     
    +const char* QU::_cudaMalloc_OOM_NOTES = R"( ;
    +QU::_cudaMalloc_OOM_NOTES
    +==========================
     
    +When running with debug arrays, such as the record array, enabled
    +it is necessary to set max_slot to something reasonable, otherwise with the 
    +default max_slot of zero, it gets set to a high value (eg M197 with 24GB) 
    +appropriate for production running with the available VRAM. 
     
    +One million is typically reasonable for debugging:: 
    +
    +   export OPTICKS_MAX_SLOT=M1
    +
    +)" ;
     
     void QU::_cudaMalloc( void** p2p, size_t size, const char* label )
     {
    @@ -259,6 +271,8 @@ void QU::_cudaMalloc( void** p2p, size_t size, const char* label )
                 sdirectory::MakeDirs(out,0); 
                 LOG(error) << "save salloc record to " << out ; 
                 alloc->save(out) ; 
    +
    +            ss << _cudaMalloc_OOM_NOTES  ; 
             }
             else
             {
    P[blyth@localhost opticks]$ 
     


