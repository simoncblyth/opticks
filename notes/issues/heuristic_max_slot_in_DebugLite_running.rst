heuristic_max_slot_in_DebugLite_running
===========================================


::

    [salloc.meta
    evt.max_curand:1000000000
    evt.max_slot:262000000
    evt.max_photon:1000000000
    evt.num_photon:29823
    evt.max_curand/M:1000
    evt.max_slot/M:262
    evt.max_photon/M:1000
    evt.num_photon/M:0
    evt.max_record:32
    evt.max_rec:0
    evt.max_seq:1
    evt.max_prd:0
    evt.max_tag:0
    evt.max_flat:0
    evt.num_record:954336
    evt.num_rec:0
    evt.num_seq:29823
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
         [            240          60           4           0]       0.00       0.00 QProp::upload/pp
         [             16           1          16           0]       0.00       0.00 QProp::upload/d_prop
         [            240          60           4           0]       0.00       0.00 QProp::upload/pp
         [             16           1          16           0]       0.00       0.00 QProp::upload/d_prop
         [             48          12           4           0]       0.00       0.00 QPMT::init_thickness/d_thickness
         [         140896       35224           4           0]       0.00       0.00 QPMT::init_lcqs/d_lcqs
         [             56           1          56           0]       0.00       0.00 QPMT::init/d_pmt
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             96           1          96           0]       0.00       0.00 QMultiFilm::uploadMultifilmlut
         [            256           1         256           0]       0.00       0.00 QEvent::QEvent/sevent
         [             64           1          64           0]       0.00       0.00 QSim::init.sim
         [        8294400     2073600           4           0]       0.01       0.00 Frame::DeviceAllo:num_pixels
         [      960000000    10000000          96           0]       0.96       0.34 QEvent::setGenstep/device_alloc_genstep_and_seed:quad6
         [     4000000000  1000000000           4           0]       4.00       1.41 QEvent::setGenstep/device_alloc_genstep_and_seed:int seed
         [    16768000000   262000000          64           0]      16.77       5.92 QEvent::device_alloc_photon/max_slot*sizeof(sphoton)
         [   261698093056  4089032704          64           0]     261.70      92.33 max_slot*max_record*sizeof(sphoton)

     tot     283434534408                                          283.43
    ]salloc::desc


::

    1004 void QEvent::device_alloc_photon()
    1005 {
    1006     LOG_IF(info, LIFECYCLE) ;
    1007     SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
    1008 
    1009     LOG(LEVEL)
    1010         << " evt.max_slot   " << evt->max_slot
    1011         << " evt.max_photon " << evt->max_photon
    1012         << " evt.num_photon " << evt->num_photon
    1013 #ifndef PRODUCTION
    1014         << " evt.num_record " << evt->num_record
    1015         << " evt.num_rec    " << evt->num_rec
    1016         << " evt.num_seq    " << evt->num_seq
    1017         << " evt.num_prd    " << evt->num_prd
    1018         << " evt.num_tag    " << evt->num_tag
    1019         << " evt.num_flat   " << evt->num_flat
    1020 #endif
    1021         ;
    1022 
    1023     evt->photon  = evt->max_slot > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot, "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)" ) : nullptr ;
    1024 
    1025 #ifndef PRODUCTION
    1026     evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot * evt->max_record, "max_slot*max_record*sizeof(sphoton)" ) : nullptr ;
    1027     evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_slot * evt->max_rec   , "max_slot*max_rec*sizeof(srec)"    ) : nullptr ;
    1028     evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_slot * evt->max_prd   , "max_slot*max_prd*sizeof(quad2)"    ) : nullptr ;
    1029     evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_slot                  , "max_slot*sizeof(sseq)"    ) : nullptr ;
    1030     evt->tag     = evt->max_tag   == 1 ? QU::device_alloc_zero<stag>(    evt->max_slot                  , "max_slot*sizeof(stag)"    ) : nullptr ;
    1031     evt->flat    = evt->max_flat  == 1 ? QU::device_alloc_zero<sflat>(   evt->max_slot                  , "max_slot*sizeof(sflat)"   ) : nullptr ;
    1032 #endif
    1033 
    1034     LOG(LEVEL) << desc() ;
    1035     LOG(LEVEL) << desc_alloc() ;
    1036 }
    1037 




    336 template<typename T>
    337 T* QU::device_alloc_zero(unsigned num_items, const char* label)
    338 {
    339     size_t size = num_items*sizeof(T) ;
    340 
    341     LOG(LEVEL)
    342         << " num_items " << std::setw(10) << num_items
    343         << " size " << std::setw(10) << size
    344         << " label " << std::setw(15) << label
    345         ;
    346 
    347     LOG_IF(info, MEMCHECK)
    348         << " num_items " << std::setw(10) << num_items
    349         << " size " << std::setw(10) << size
    350         << " label " << std::setw(15) << label
    351         ;
    352 
    353 
    354     alloc_add( label, size, num_items, sizeof(T), 0 ) ;
    355 
    356     T* d ;
    357     _cudaMalloc( reinterpret_cast<void**>( &d ), size, label );
    358 
    359     int value = 0 ;
    360     QUDA_CHECK( cudaMemset(d, value, size ));
    361 
    362     return d ;
    363 }


    046 void QU::alloc_add(const char* label, uint64_t size, uint64_t num_items, uint64_t sizeof_item, uint64_t spare) // static
     47 {
     48    if(!alloc) alloc = SEventConfig::ALLOC ;
     49    if(alloc ) alloc->add(label, size, num_items, sizeof_item , spare);
     50 }
     51 





::

    max_record:32
    sizeof(sphoton):4*4*sizeof(float) = 4*4*4 = 64 bytes

    32*64 = 2048 bytes
 
    262e6 * 2048 

     



