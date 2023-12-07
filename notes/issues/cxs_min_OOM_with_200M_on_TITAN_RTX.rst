cxs_min_OOM_with_200M_on_TITAN_RTX
====================================

::

    ./cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2023-12-07 15:22:05.923  923098017 : [./cxs_min.sh 
    2023-12-07 15:24:06.826 INFO  [43088] [QSim::simulate@366]  eventID 0 dt   72.123257 ph  200000000
    2023-12-07 15:24:06.855 ERROR [43088] [QU::_cudaMalloc@230] save salloc record to /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (QEvent::gatherHit_:sphoton ) failed with error: 'out of memory' (/home/blyth/junotop/opticks/qudarap/QU.cc:223)
    salloc::desc alloc.size 23 label.size 23
    salloc.meta
    evt.max_photon:200000000
    evt.max_record:0
    evt.max_rec:0
    evt.max_seq:0
    evt.max_prd:0
    evt.max_tag:0
    evt.max_flat:0
    evt.num_photon:200000000
    evt.num_record:0
    evt.num_rec:0
    evt.num_seq:0
    evt.num_prd:0
    evt.num_tag:0
    evt.num_flat:0


         [           size   num_items sizeof_item       spare]    size_GB    percent label
         [        (bytes)                                    ]   size/1e9            

         [              4           1           4           0]       0.00       0.00 QBase::init/d_base
         [     9600000000   200000000          48           0]       9.60      36.44 QRng::upload/rng_states
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
         [        8294400     2073600           4           0]       0.01       0.03 Frame::DeviceAllo:num_pixels
         [      288000000     3000000          96           0]       0.29       1.09 QEvent::setGenstep/device_alloc_genstep_and_seed:quad6
         [      800000000   200000000           4           0]       0.80       3.04 QEvent::setGenstep/device_alloc_genstep_and_seed:int seed
         [    12800000000   200000000          64           0]      12.80      48.58 QEvent::device_alloc_photon/max_photon*sizeof(sphoton)
         [     2851617152    44556518          64           0]       2.85      10.82 QEvent::gatherHit_:sphoton

     tot      26348057468                                           26.35

    ./cxs_min.sh: line 253: 43088 Aborted                 (core dumped) $bin
    ./cxs_min.sh run error
    N[blyth@localhost opticks]$ 



