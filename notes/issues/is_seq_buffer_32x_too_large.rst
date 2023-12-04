is_seq_buffer_32x_too_large
============================

Related
---------

* :doc:`OPTICKS_MAX_BOUNCE_scanning`


Overview
----------

While thinking about OPTICKS_MAX_BOUNCE scanning notice overlarge allocation
for seq. 

SEventConfig.cc::

    722     else if(IsStandardFullDebug())
    723     {
    724         SEventConfig::SetMaxRecord(max_bounce+1);
    725         SEventConfig::SetMaxRec(max_bounce+1);
    726         SEventConfig::SetMaxSeq(max_bounce+1);  // HUH: seq is photon level thing ? 
    727         SEventConfig::SetMaxPrd(max_bounce+1);
    728         SEventConfig::SetMaxAux(max_bounce+1);
    729 
    730         // since moved to compound sflat/stag so MaxFlat/MaxTag should now either be 0 or 1, nothing else  
    731         SEventConfig::SetMaxTag(1);
    732         SEventConfig::SetMaxFlat(1);
    733         SEventConfig::SetMaxSup(1);
    734 
    735         SetComp() ;   // comp set based on Max values   
    736     }


QEvent.cc::

    930     evt->seq     = evt->max_seq    > 0 ? QU::device_alloc_zero<sseq>(    evt->max_photon * evt->max_seq   , "max_photon*max_seq*sizeof(sseq)"    ) : nullptr ;

sctx.h::

    129 SCTX_METHOD void sctx::point(int bounce)
    130 {
    131     if(evt->record && bounce < evt->max_record) evt->record[evt->max_record*idx+bounce] = p ;
    132     if(evt->rec    && bounce < evt->max_rec)    evt->add_rec( rec, idx, bounce, p );    // this copies into evt->rec array 
    133     if(evt->seq    && bounce < evt->max_seq)    seq.add_nibble( bounce, p.flag(), p.boundary() );
    134     if(evt->aux    && bounce < evt->max_aux)    evt->aux[evt->max_aux*idx+bounce] = aux ;
    135 }

    // bounce is the slot pointing at the nibble to write within the sseq NSEQ ulls

::

     601 void QEvent::gatherSeq(NP* seq) const
     602 {
     603     bool has_seq = hasSeq();
     604     if(!has_seq) return ;
     605     LOG(LEVEL) << "[ evt.num_seq " << evt->num_seq << " seq.sstr " << seq->sstr() << " evt.seq " << evt->seq ;
     606     assert( seq->has_shape(evt->num_seq, 2) );
     607     QU::copy_device_to_host<sseq>( (sseq*)seq->bytes(), evt->seq, evt->num_seq );
     608     LOG(LEVEL) << "] evt.num_seq " << evt->num_seq  ;
     609 }

::

    2084 void SEvt::setNumPhoton(unsigned num_photon)
    2085 {
    2086     //LOG_IF(info, LIFECYCLE) << id() << " num_photon " << num_photon ; 
    2087     bool num_photon_allowed = int(num_photon) <= evt->max_photon ;
    2088     const int M = 1000000 ;
    2089 
    2090     LOG_IF(fatal, !num_photon_allowed) << " num_photon/M " << num_photon/M << " evt.max_photon/M " << evt->max_photon/M ;
    2091     assert( num_photon_allowed );
    2092 
    2093     evt->num_photon = num_photon ;
    2094     evt->num_seq    = evt->max_seq   > 0 ? evt->num_photon : 0 ;
    2095     evt->num_tag    = evt->max_tag  == 1 ? evt->num_photon : 0 ;
    2096     evt->num_flat   = evt->max_flat == 1 ? evt->num_photon : 0 ;
    2097     evt->num_sup    = evt->max_sup   > 0 ? evt->num_photon : 0 ;
    2098 





is seq array (max_bounce+1) eg 32 too large ? 
--------------------------------------------------

::

    epsilon:opticks blyth$ opticks-f max_seq
    ./sysrap/SEventConfig.hh:    static void SetMaxSeq(    int max_seq); 
    ./sysrap/SEventConfig.cc:void SEventConfig::SetMaxSeq(    int max_seq){     _MaxSeq     = max_seq     ; Check() ; }
    ./sysrap/sctx.h:    if(evt->seq    && bounce < evt->max_seq)    seq.add_nibble( bounce, p.flag(), p.boundary() );  
    ./sysrap/sevent.h:    int      max_seq     ; // eg: 16  seqhis/seqbnd
    ./sysrap/sevent.h:    max_seq      = SEventConfig::MaxSeq()  ;     // seqhis 
    ./sysrap/sevent.h:        << " evt.max_seq       " << std::setw(w) << max_seq      << std::endl 
    ./sysrap/sevent.h:        << std::setw(20) << " max_seq "         << std::setw(7) << max_seq 
    ./sysrap/sevent.h:   cfg.q1.u.z = max_seq ; 
    ./sysrap/sevent.h:    NP::SetMeta<uint64_t>(meta,"evt.max_seq", max_seq); 
    ./sysrap/SEvt.cc:    evt->num_seq    = evt->max_seq   > 0 ? evt->num_photon : 0 ;
    ./sysrap/SEvt.cc:    if( evt->seq && prior < evt->max_seq )
    ./sysrap/SEvt.cc:        << " evt.max_seq    " << evt->max_seq
    ./qudarap/QEvent.cc:    evt->seq     = evt->max_seq    > 0 ? QU::device_alloc_zero<sseq>(    evt->max_photon * evt->max_seq   , "max_photon*max_seq*sizeof(sseq)"    ) : nullptr ; 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 



QEvent=INFO salloc dumping
-------------------------------

::

    2023-12-04 11:30:17.048 INFO  [245043] [QEvent::device_alloc_photon@953] [QEvent::desc_alloc 
    salloc::desc
    salloc::desc alloc.size 28 label.size 28
    salloc.meta
    evt.max_photon:1000000
    evt.max_record:32
    evt.max_rec:32
    evt.max_seq:32    ## NOW RESTRICT TO 0 OR 1 
    evt.max_prd:32
    evt.max_tag:1
    evt.max_flat:1
    evt.num_photon:100000
    evt.num_record:3200000
    evt.num_rec:3200000
    evt.num_seq:100000
    evt.num_prd:3200000
    evt.num_tag:100000
    evt.num_flat:100000


         [           size   num_items sizeof_item       spare]    size_GB    percent label
         [        (bytes)                                    ]   size/1e9            

         [              4           1           4           0]       0.00       0.00 QBase::init/d_base
         [      144000000     3000000          48           0]       0.14       2.66 QRng::upload/rng_states
         [             16           1          16           0]       0.00       0.00 QRng::upload/d_qr
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
         [            240           1         240           0]       0.00       0.00 QEvent::QEvent/sevent
         [             64           1          64           0]       0.00       0.00 QSim::init.sim
         [        8294400     2073600           4           0]       0.01       0.15 Frame::DeviceAllo:num_pixels
         [      288000000     3000000          96           0]       0.29       5.33 QEvent::setGenstep/device_alloc_genstep_and_seed:quad6
         [        4000000     1000000           4           0]       0.00       0.07 QEvent::setGenstep/device_alloc_genstep_and_seed:int seed
         [       64000000     1000000          64           0]       0.06       1.18 QEvent::device_alloc_photon/max_photon*sizeof(sphoton)
         [     2048000000    32000000          64           0]       2.05      37.89 max_photon*max_record*sizeof(sphoton)
         [      512000000    32000000          16           0]       0.51       9.47 max_photon*max_rec*sizeof(srec)

         [     1024000000    32000000          32           0]       1.02      18.95 max_photon*max_seq*sizeof(sseq)

         [     1024000000    32000000          32           0]       1.02      18.95 max_photon*max_prd*sizeof(quad2)
         [       32000000     1000000          32           0]       0.03       0.59 max_photon*sizeof(stag)
         [      256000000     1000000         256           0]       0.26       4.74 max_photon*sizeof(sflat)

     tot       5404440316                                            5.40

    ]QEvent::desc_alloc 



The downloaded sizes are less than the device buffer sizes by design.
As the device buffers are based on max photon::

    N[blyth@localhost ALL99]$ du -hs p0??/seq.npy
    3.1M	p001/seq.npy
    6.2M	p002/seq.npy
    9.2M	p003/seq.npy
    13M	p004/seq.npy
    16M	p005/seq.npy
    19M	p006/seq.npy
    22M	p007/seq.npy
    25M	p008/seq.npy
    28M	p009/seq.npy
    31M	p010/seq.npy

But even when using max photon here of 1M only get to 32MB 
compared to buffer size of 1GB (32x bigger). 


    N[blyth@localhost p010]$ l
    total 3388992
          4 -rw-rw-r--.  1 blyth blyth        717 Dec  4 11:06 NPFold_meta.txt
          0 -rw-rw-r--.  1 blyth blyth          0 Dec  4 11:06 NPFold_names.txt
          4 -rw-rw-r--.  1 blyth blyth        113 Dec  4 11:06 sframe_meta.txt
          4 -rw-rw-r--.  1 blyth blyth        384 Dec  4 11:06 sframe.npy
     250004 -rw-rw-r--.  1 blyth blyth  256000128 Dec  4 11:06 flat.npy
          4 -rw-rw-r--.  1 blyth blyth        256 Dec  4 11:06 domain.npy
      31252 -rw-rw-r--.  1 blyth blyth   32000128 Dec  4 11:06 tag.npy
      13940 -rw-rw-r--.  1 blyth blyth   14274432 Dec  4 11:06 hit.npy
    1000004 -rw-rw-r--.  1 blyth blyth 1024000144 Dec  4 11:06 prd.npy
      31252 -rw-rw-r--.  1 blyth blyth   32000128 Dec  4 11:06 seq.npy          ## 32 MB
          4 -rw-rw-r--.  1 blyth blyth         36 Dec  4 11:06 record_meta.txt
    2000004 -rw-rw-r--.  1 blyth blyth 2048000144 Dec  4 11:06 record.npy
          4 -rw-rw-r--.  1 blyth blyth         74 Dec  4 11:06 NPFold_index.txt
      62504 -rw-rw-r--.  1 blyth blyth   64000128 Dec  4 11:06 photon.npy
          0 drwxrwxr-x. 12 blyth blyth        187 Dec  4 11:03 ..
          4 drwxr-xr-x.  2 blyth blyth       4096 Nov 29 20:51 .
          4 -rw-rw-r--.  1 blyth blyth        224 Nov 29 20:51 genstep.npy
    N[blyth@localhost p010]$ 



