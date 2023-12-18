okjob_GPU_memory_leak
=======================

Overview
----------

Most likely culprits, as more dynamic allocation handling are:

1. hits
2. gensteps 


* okjob.sh gun:1 is leaking steadily at 0.003 [GB/s] measured with smonitor.sh, 
  this has plenty of both gensteps and hits 

* "TEST=large_scan cxs_min.sh" torch running does not leak, this has millions of hits but only one small genstep 
  suggesting hit handling is OK



0.003 GB/s is 10GB in ~55min::

    In [1]: 10/0.003
    Out[1]: 3333.3333333333335

    In [2]: (10/0.003)/60 
    Out[2]: 55.55555555555556




::

   MEMCHECK=1 BP="cudaMalloc cudaFree" ~/o/cxs_min.sh 


gdb investigate
------------------

* 53 cudaMalloc to first setGenstep cudaMalloc

::

    In [2]: 6*4*4*3000000   ## 3M gensteps
    Out[2]: 288000000





::

    (gdb) bt
    #0  0x00007ffff7586100 in cudaMalloc () from /data/blyth/junotop/ExternalLibs/opticks/head/lib/../lib64/libQUDARap.so
    #1  0x00007ffff74eb5b2 in QU::_cudaMalloc (p2p=0x7fffffff0040, size=288000000, 
        label=0x7ffff75b7aa8 "QEvent::setGenstep/device_alloc_genstep_and_seed:quad6") at /home/blyth/junotop/opticks/qudarap/QU.cc:219
    #2  0x00007ffff74f8383 in QU::device_alloc<quad6> (num_items=3000000, 
        label=0x7ffff75b7aa8 "QEvent::setGenstep/device_alloc_genstep_and_seed:quad6") at /home/blyth/junotop/opticks/qudarap/QU.cc:256
    #3  0x00007ffff74de61a in QEvent::device_alloc_genstep_and_seed (this=0xad900f0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:352
    #4  0x00007ffff74de018 in QEvent::setGenstepUpload (this=0xad900f0, qq=0xc94bbe0, num_genstep=140)
        at /home/blyth/junotop/opticks/qudarap/QEvent.cc:284
    #5  0x00007ffff74ddc34 in QEvent::setGenstepUpload_NP (this=0xad900f0, gs_=0xc8d5950) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:250
    #6  0x00007ffff74dd8ef in QEvent::setGenstep (this=0xad900f0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:196
    #7  0x00007ffff74a1b4b in QSim::simulate (this=0xad90040, eventID=0, reset_=true) at /home/blyth/junotop/opticks/qudarap/QSim.cc:357
    #8  0x00007ffff7e59897 in CSGOptiX::simulate (this=0xad9ecc0, eventID=0) at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:674
    #9  0x00007ffff7e56583 in CSGOptiX::SimulateMain () at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:180
    #10 0x0000000000405b15 in main (argc=1, argv=0x7fffffff21f8) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) i b 




::

    BP="cudaMalloc cudaFree" ~/o/cxs_min.sh 


PROGRESS : managed to get cxs_min.sh to leak using gensteps from okjob.sh
---------------------------------------------------------------------------

Using real input genstep from okjob.sh within cxs_min.sh succeeds to leak
Thats great, because cxs_min.sh can boot in <2s::

    TEST=input_genstep ~/o/cxs_min.sh  


TEST=setGenstep_many ~/o/qudarap/tests/QEventTest.sh   ## NO LEAK
---------------------------------------------------------------------

compute sanitizer
------------------

* https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#leak-checking

cuda-memcheck
----------------

Probably not very useful as I dont care about leaking 
initialization things like geometry and fixed stuff. 
Its only the event by event increase thats problematic.


thrust::reserve issue
-----------------------

* https://github.com/NVIDIA/thrust/issues/1443


BP=cudaMalloc LOG=1 ~/j/okjob.sh 
------------------------------------

Breaking in all cudaMalloc shows that after initialization allocs the only 
event by event allocs are from two sources:

1. QEvent::setGenstep/.../QEvent_count_genstep_photons_and_fill_seed_buffer   from thrust 
2. QEvent::gatherHit/.../SU::count_if_sphoton   from thrust::detail::temporary_allocator
   QEvent::gatherHit/.../QU::device_alloc<sphoton> 


So suspicion falls on : QEvent_count_genstep_photons_and_fill_seed_buffer





    Thread 1 "python" hit Breakpoint 1, 0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    (gdb) bt
    #0  0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #1  0x00007fffc822e133 in thrust::detail::temporary_allocator<unsigned char, thrust::cuda_cub::tag>::allocate(unsigned long) [clone .isra.0] ()
       from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #2  0x00007fffc8236ea0 in int thrust::cuda_cub::reduce_n<thrust::cuda_cub::tag, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, long, int, thrust::plus<int> >(thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag>&, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, long, int, thrust::plus<int>) [clone .isra.0] () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #3  0x00007fffc8237734 in thrust::iterator_traits<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> > >::value_type thrust::reduce<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> > >(thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >) () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #4  0x00007fffc822e426 in QEvent_count_genstep_photons_and_fill_seed_buffer () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #5  0x00007fffc81d71ee in QEvent::count_genstep_photons_and_fill_seed_buffer (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:513
    #6  0x00007fffc81d6231 in QEvent::setGenstepUpload (this=0x1c19cab0, qq=0xa58ce810, num_genstep=140) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:310
    #7  0x00007fffc81d5c34 in QEvent::setGenstepUpload_NP (this=0x1c19cab0, gs_=0xa58c1060) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:250
    #8  0x00007fffc81d58ef in QEvent::setGenstep (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:196


    Thread 1 "python" hit Breakpoint 1, 0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    (gdb) bt
    #0  0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #1  0x00007fffc82352e8 in void iexpand<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::device_ptr<int> >(thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::device_ptr<int>, thrust::device_ptr<int>) () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #2  0x00007fffc822e487 in QEvent_count_genstep_photons_and_fill_seed_buffer () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #3  0x00007fffc81d71ee in QEvent::count_genstep_photons_and_fill_seed_buffer (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:513
    #4  0x00007fffc81d6231 in QEvent::setGenstepUpload (this=0x1c19cab0, qq=0xa58ce810, num_genstep=140) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:310
    #5  0x00007fffc81d5c34 in QEvent::setGenstepUpload_NP (this=0x1c19cab0, gs_=0xa58c1060) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:250
    #6  0x00007fffc81d58ef in QEvent::setGenstep (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:196
    #7  0x00007fffc8199b4b in QSim::simulate (this=0x1c19ca00, eventID=0, reset_=false) at /home/blyth/junotop/opticks/qudarap/QSim.cc:357
    #8  0x00007fffc8eb8b6c in G4CXOpticks::simulate (this=0xa178430, eventID=0, reset_=false) at /home/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:377



    (gdb) bt
    #0  0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #1  0x00007fffc822e133 in thrust::detail::temporary_allocator<unsigned char, thrust::cuda_cub::tag>::allocate(unsigned long) [clone .isra.0] ()
       from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #2  0x00007fffc82341ab in thrust::detail::normal_iterator<thrust::device_ptr<long> > thrust::cuda_cub::detail::exclusive_scan_n_impl<thrust::cuda_cub::tag, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, long, thrust::detail::normal_iterator<thrust::device_ptr<long> >, int, thrust::plus<void> >(thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag>&, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, long, thrust::detail::normal_iterator<thrust::device_ptr<long> >, int, thrust::plus<void>) [clone .isra.0] () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #3  0x00007fffc8234f18 in thrust::detail::normal_iterator<thrust::device_ptr<long> > thrust::exclusive_scan<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::detail::normal_iterator<thrust::device_ptr<long> > >(thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::detail::normal_iterator<thrust::device_ptr<long> >) () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #4  0x00007fffc8234ff3 in void iexpand<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::device_ptr<int> >(thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::device_ptr<int>, thrust::device_ptr<int>) () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #5  0x00007fffc822e487 in QEvent_count_genstep_photons_and_fill_seed_buffer () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #6  0x00007fffc81d71ee in QEvent::count_genstep_photons_and_fill_seed_buffer (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:513
    #7  0x00007fffc81d6231 in QEvent::setGenstepUpload (this=0x1c19cab0, qq=0xa58ce810, num_genstep=140) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:310
    #8  0x00007fffc81d5c34 in QEvent::setGenstepUpload_NP (this=0x1c19cab0, gs_=0xa58c1060) at /home/blyth/junotop/opticks/qudar



    Thread 1 "python" hit Breakpoint 1, 0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    (gdb) bt
    #0  0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #1  0x00007fffc822e133 in thrust::detail::temporary_allocator<unsigned char, thrust::cuda_cub::tag>::allocate(unsigned long) [clone .isra.0] ()
       from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #2  0x00007fffc82351e4 in void iexpand<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::device_ptr<int> >(thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::device_ptr<int>, thrust::device_ptr<int>) () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #3  0x00007fffc822e487 in QEvent_count_genstep_photons_and_fill_seed_buffer () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #4  0x00007fffc81d71ee in QEvent::count_genstep_photons_and_fill_seed_buffer (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:513
    #5  0x00007fffc81d6231 in QEvent::setGenstepUpload (this=0x1c19cab0, qq=0xa58ce810, num_genstep=140) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:310
    #6  0x00007fffc81d5c34 in QEvent::setGenstepUpload_NP (this=0x1c19cab0, gs_=0xa58c1060) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:250
    #7  0x00007fffc81d58ef in QEvent::setGenstep (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:196
    #8  0x00007fffc8199b4b in QSim::simulate (this=0x1c19ca00, eventID=0, reset_=false) at /home/blyth/junotop/opticks/qudarap/QSim.cc:357
    #9  0x00007fffc8eb8b6c in G4CXOpticks::simulate (this=0xa178430, eventID=0, reset_=false) at /home/blyth/junotop/opticks/g4cx/G4CX




    Thread 1 "python" hit Breakpoint 1, 0x00007fffc7ffb920 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libSysRap.so
    (gdb) bt
    #0  0x00007fffc7ffb920 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libSysRap.so
    #1  0x00007fffc7facaa3 in thrust::detail::temporary_allocator<unsigned char, thrust::cuda_cub::tag>::allocate(unsigned long) [clone .isra.0] ()
       from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libSysRap.so
    #2  0x00007fffc7fad088 in long thrust::cuda_cub::reduce_n<thrust::cuda_cub::tag, thrust::cuda_cub::transform_input_iterator_t<long, thrust::device_ptr<sphoton const>, sphoton_selector>, long, long, thrust::plus<long> >(thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag>&, thrust::cuda_cub::transform_input_iterator_t<long, thrust::device_ptr<sphoton const>, sphoton_selector>, long, long, thrust::plus<long>) [clone .isra.0] () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libSysRap.so
    #3  0x00007fffc7fad789 in SU::count_if_sphoton(sphoton const*, unsigned int, sphoton_selector const&) () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libSysRap.so
    #4  0x00007fffc81d98d9 in QEvent::gatherHit (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:779
    #5  0x00007fffc81da2f4 in QEvent::gatherComponent_ (this=0x1c19cab0, cmp=256) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:860
    #6  0x00007fffc81da00f in QEvent::gatherComponent (this=0x1c19cab0, cmp=256) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:838
    #7  0x00007fffc7f3b90a in SEvt::gather_components (this=0x13bbd720) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:3490
    #8  0x00007fffc7f3c4de in SEvt::gather (this=0x13bbd720) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:3576
    #9  0x00007fffc8199ce1 in QSim::simulate (this=0x1c19ca00, eventID=0, reset_=false) at /home/blyth/junotop/opticks/qudarap/QSim.cc




    hread 1 "python" hit Breakpoint 1, 0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    (gdb) bt
    #0  0x00007fffc827e100 in cudaMalloc () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #1  0x00007fffc822e133 in thrust::detail::temporary_allocator<unsigned char, thrust::cuda_cub::tag>::allocate(unsigned long) [clone .isra.0] ()
       from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #2  0x00007fffc8236ea0 in int thrust::cuda_cub::reduce_n<thrust::cuda_cub::tag, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, long, int, thrust::plus<int> >(thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag>&, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, long, int, thrust::plus<int>) [clone .isra.0] () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #3  0x00007fffc8237734 in thrust::iterator_traits<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> > >::value_type thrust::reduce<thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> > >(thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >, thrust::permutation_iterator<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::transform_iterator<strided_range<thrust::detail::normal_iterator<thrust::device_ptr<int> > >::stride_functor, thrust::counting_iterator<long, thrust::use_default, thrust::use_default, thrust::use_default>, thrust::use_default, thrust::use_default> >) () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #4  0x00007fffc822e426 in QEvent_count_genstep_photons_and_fill_seed_buffer () from /home/blyth/junotop/ExternalLibs/opticks/head/lib64/libQUDARap.so
    #5  0x00007fffc81d71ee in QEvent::count_genstep_photons_and_fill_seed_buffer (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:513
    #6  0x00007fffc81d6231 in QEvent::setGenstepUpload (this=0x1c19cab0, qq=0xb1e109e0, num_genstep=117) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:310
    #7  0x00007fffc81d5c34 in QEvent::setGenstepUpload_NP (this=0x1c19cab0, gs_=0xa58ac6e0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:250
    #8  0x00007fffc81d58ef in QEvent::setGenstep (this=0x1c19cab0) at /home/blyth/junotop/opticks/qudarap/QEvent.cc:196





cuda-memcheck
--------------



nvprof
--------

* https://docs.nvidia.com/cuda/profiler-users-guide/index.html


QEvent__LIFECYCLE check
-------------------------

::

    ~/j/okjob.sh 



cxs_min.sh : NOT LEAKING 
---------------------------

Workstation::

    ~/o/sysrap/smonitor.sh build
    ~/o/sysrap/smonitor.sh run

    TEST=large_scan ~/o/cxs_min.sh 

    CTRL-C the smonitor


::

    .
     [167.325  12.735]
     [168.327  12.735]
     [169.328  12.735]
     [170.332  12.735]
     [171.334  12.735]
     [172.336  12.735]
     [173.338  12.735]]
    dmem      0.002  (usedGpuMemory_GB[sel][-1]-usedGpuMemory_GB[sel][0]) 
    dt      153.299  (t[sel][-1]-t[sel][0]) 
    dmem/dt       0.000  
    smonitor.sh device 0 total_GB 25.8 pid 96770 
    line fit:  slope      0.001 [GB/s] intercept     12.702 


QEvent_Lifecycle_Test.sh : NOT LEAKING
------------------------------------------

::

    ~/o/qudarap/tests/QEvent_Lifecycle_Test.sh 



okjob.sh leaking at 0.003 GB/s (from smonitor.sh)
----------------------------------------------------

* tried changing to event mode Nothing : but thats too bit a change for comparable numbers 

::

    np.c_[t[sel], usedGpuMemory_GB[sel]]
    [[128.246   1.345]
     [129.247   1.35 ]
     [130.249   1.354]
     ...
     [166.322   1.478]
     [167.324   1.481]
     [168.326   1.481]
     [169.328   1.483]
     [170.329   1.483]]
    dmem      0.137  (usedGpuMemory_GB[sel][-1]-usedGpuMemory_GB[sel][0]) 
    dt       42.083  (t[sel][-1]-t[sel][0]) 
    dmem/dt       0.003  
    smonitor.sh device 0 total_GB 25.8 pid 280674 
    line fit:  slope      0.003 [GB/s] intercept      0.907 


     [166.329   1.481]
     [167.331   1.483]
     [168.333   1.483]]
    dmem      0.133  (usedGpuMemory_GB[sel][-1]-usedGpuMemory_GB[sel][0]) 
    dt       41.083  (t[sel][-1]-t[sel][0]) 
    dmem/dt       0.003  
    smonitor.sh device 0 total_GB 25.8 pid 212028 
    line fit:  slope      0.003 [GB/s] intercept      0.918 




nvidia-smi monitoring : very rough eyeballing
-------------------------------------------------

During 1000 event run monitor with::

    nvidia-smi -lms 500    # every half second 



starts flat at 941Mib::


    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     13888      G   /usr/bin/X                         24MiB |
    |    0   N/A  N/A     15789      G   /usr/bin/gnome-shell              112MiB |
    |    0   N/A  N/A     16775      G   /usr/bin/X                        129MiB |
    |    0   N/A  N/A     23246      C   python                            941MiB |
    |    0   N/A  N/A    352750      G   /usr/bin/gnome-shell               14MiB |
    +-----------------------------------------------------------------------------+

Jumps to 1283MiB::

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     13888      G   /usr/bin/X                         24MiB |
    |    0   N/A  N/A     15789      G   /usr/bin/gnome-shell              112MiB |
    |    0   N/A  N/A     16775      G   /usr/bin/X                        129MiB |
    |    0   N/A  N/A     23246      C   python                           1283MiB |
    |    0   N/A  N/A    352750      G   /usr/bin/gnome-shell               14MiB |
    +-----------------------------------------------------------------------------+

Then proceeds steadily upwards ending after 1000 launches at 1414MiB::

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     13888      G   /usr/bin/X                         24MiB |
    |    0   N/A  N/A     15789      G   /usr/bin/gnome-shell              112MiB |
    |    0   N/A  N/A     16775      G   /usr/bin/X                        129MiB |
    |    0   N/A  N/A     23246      C   python                           1414MiB |
    |    0   N/A  N/A    352750      G   /usr/bin/gnome-shell               15MiB |
    +-----------------------------------------------------------------------------+


* 1414-1283 

::

    In [2]: (1414-1283)/1000.
    Out[2]: 0.131


Leaking about 0.1 MB per launch 



pynvml
----------

Install pynvml with conda::

    N[blyth@localhost nvml_py]$ ./moni.py 
    devcount:2 
    handle:<pynvml.nvml.LP_struct_c_nvmlDevice_t object at 0x7fc05499d440>
    {'pid': 226283, 'usedGpuMemory': 986710016, 'gpuInstanceId': 4294967295, 'computeInstanceId': 4294967295}
    pid 226283 using 986710016 bytes of memory on device 0.
    handle:<pynvml.nvml.LP_struct_c_nvmlDevice_t object at 0x7fc05499cf80>


::

    N[blyth@localhost nvml_py]$ cat ~/nvml_py/moni.py 
    #!/usr/bin/env python

    import pynvml

    pynvml.nvmlInit()

    devcount = pynvml.nvmlDeviceGetCount()
    print("devcount:%d " % devcount )

    for dev_id in range(devcount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        print("handle:%s" % handle) 

        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):

            print(proc)
            print(
                "pid %d using %d bytes of memory on device %d."
                % (proc.pid, proc.usedGpuMemory, dev_id)
            )



    N[blyth@localhost nvml_py]$ 



