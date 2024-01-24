okjob_GPU_memory_leak
=======================


FIXED : was caused by using separate CUDA stream for every launch
--------------------------------------------------------------------

::

    1027 #if OPTIX_VERSION < 70000
    1028     assert( width <= 1000000 );
    1029     six->launch(width, height, depth );
    1030 #else
    1031     if(DEBUG_SKIP_LAUNCH == false)
    1032     {
    1033         CUdeviceptr d_param = (CUdeviceptr)Params::d_param ; ;
    1034         assert( d_param && "must alloc and upload params before launch");
    1035 
    1036         /*
    1037         // this way leaking 14kb for every launch  : see 
    1038         //
    1039         //       ~/opticks/notes/issues/okjob_GPU_memory_leak.rst
    1040         //       ~/opticks/CSGOptiX/cxs_min_igs.sh 
    1041         // 
    1042         CUstream stream ;
    1043         CUDA_CHECK( cudaStreamCreate( &stream ) );
    1044         OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    1045         */
    1046 
    1047         // Using the default stream seems to avoid 14k VRAM leak at every launch. 
    1048         // Does that mean every launch gets to use the same single default stream ?  
    1049         CUstream stream = 0 ;
    1050         OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    1051 
    1052         CUDA_SYNC_CHECK();
    1053         // see CSG/CUDA_CHECK.h the CUDA_SYNC_CHECK does cudaDeviceSyncronize
    1054         // THIS LIKELY HAS LARGE PERFORMANCE IMPLICATIONS : BUT NOT EASY TO AVOID (MULTI-BUFFERING ETC..)  
    1055     }
    1056 #endif




Speeddial
-------------

::

   nvidia-smi -lms 500    # every half second  


Investigations using input gensteps
-------------------------------------

::

    ~/o/CSGOptiX/cxs_min_igs.sh


Strategy
-------------

GPU memory monitoring is coarser than CPU and done separately 
so the approach is very different to CPU where can just add more
profiling stamps to find where memory gets consumed. 

Have to adopt indirect approaches. Start by trying to get 
a QEvent test to exhibit the GPU memory leak. 




Overview
----------

Most likely culprits, as more dynamic allocation handling are:

1. hits
2. gensteps 


* okjob.sh gun:1 is leaking steadily at 0.003 [GB/s] measured with smonitor.sh, 
  this has plenty of both gensteps and hits 

* "TEST=large_scan cxs_min.sh" torch running does not leak, this has millions of hits but only one small genstep 
  suggesting hit handling is OK : so genstep handling is under suspicion



0.003 GB/s is 10GB in ~55min::

    In [1]: 10/0.003
    Out[1]: 3333.3333333333335

    In [2]: (10/0.003)/60 
    Out[2]: 55.55555555555556


::

   MEMCHECK=1 BP="cudaMalloc cudaFree" ~/o/cxs_min.sh 



DONE : implement input genstep running from a file path pattern 
-----------------------------------------------------------------

::

    193 if [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_GENSTEP" ]; then
    194 
    195     igs=$BASE/jok-tds/ALL0/A000/genstep.npy
    196     # TODO: impl handling a sequence of input genstep 
    197     export OPTICKS_INPUT_GENSTEP=$igs
    198     [ ! -f "$igs" ] && echo $BASH_SOURCE : FATAL : NO SUCH PATH : igs $igs && exit 1
    199 




review from top
-----------------

::

    G4CXOpticks::simulate
    QSim::simulate
       SEvt::beginOfEvent
       QEvent::setGenstep
       CSGOptiX::simulate_launch
       SEvt::gather
       SEvt::reset   (when reset:true)
          SEvt::endOfEvent





    
DONE : added pure opticks input genstep running 
------------------------------------------------

* adhoc look at nvidia-smi -l 1 during ~/o/CSGOptiX/cxs_min_igs.sh suggests 
  GPU leak is still apparent with input genstep running

* advantage of course is <1s init time and a lot less code being run

* with QSim__simulate_DEBUG_SKIP_LAUNCH dont see the leak, so its not from QEvent::setGenstep

  

::

     01 #!/bin/bash -l 
      2 usage(){ cat << EOU
      3 cxs_min_igs.sh
      4 ===============
      5 
      6 ::
      7 
      8    ~/o/CSGOptiX/cxs_min_igs.sh 
      9 
     10 * skipping the launch, dont see the leak : GPU mem stays 1283 MiB
     11 * with the launch, clear continuous growth from 1283 MiB across 1000 evt 
     12 * skipping the gather only (not the launch) still leaking the same
     13 
     14 EOU
     15 }



Look at OptiX SDK examples to see what I am doing differently
----------------------------------------------------------------

/Developer/OptiX_750/SDK/optixRaycasting/optixRaycasting.cpp::

    291 void launch( RaycastingState& state )
    292 {
    293     CUstream stream_1 = 0;
    294     CUstream stream_2 = 0;
    295     CUDA_CHECK( cudaStreamCreate( &stream_1 ) );
    296     CUDA_CHECK( cudaStreamCreate( &stream_2 ) );
    297 
    298     Params* d_params            = 0;
    299     Params* d_params_translated = 0;
    300     CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( Params ) ) );
    301     CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ), &state.params, sizeof( Params ),
    302                                  cudaMemcpyHostToDevice, stream_1 ) );
    303 
    304     OPTIX_CHECK( optixLaunch( state.pipeline_1, stream_1, reinterpret_cast<CUdeviceptr>( d_params ), sizeof( Params ),
    305                               &state.sbt, state.width, state.height, 1 ) );
    306 
    307     // Translated
    308     CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params_translated ), sizeof( Params ) ) );
    309     CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params_translated ), &state.params_translated,
    310                                  sizeof( Params ), cudaMemcpyHostToDevice, stream_2 ) );
    311 
    312     OPTIX_CHECK( optixLaunch( state.pipeline_2, stream_2, reinterpret_cast<CUdeviceptr>( d_params_translated ),
    313                               sizeof( Params ), &state.sbt, state.width, state.height, 1 ) );
    314 
    315     CUDA_SYNC_CHECK();
    316 
    317     CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params ) ) );
    318     CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params_translated ) ) );
    319 }


Params are cudaMalloc and cudaFree for each launch, 
but I alloc once at initialization ?



 

review the cxs_min.sh code
-----------------------------


::

     174 int CSGOptiX::SimulateMain() // static
     175 {
     176     SProf::Add("CSGOptiX__SimulateMain_HEAD");
     177     SEventConfig::SetRGModeSimulate();
     178     CSGFoundry* fd = CSGFoundry::Load();
     179     CSGOptiX* cx = CSGOptiX::Create(fd) ;
     180     for(int i=0 ; i < SEventConfig::NumEvent() ; i++) cx->simulate(i);
     181     SProf::UnsetTag();
     182     SProf::Add("CSGOptiX__SimulateMain_TAIL");
     183     SProf::Write("run_meta.txt", true ); // append:true 
     184     cx->write_Ctx_log();
     185     return 0 ;
     186 }


     669 double CSGOptiX::simulate(int eventID)
     670 {
     671     SProf::SetTag(eventID, "A%0.3d_" ) ;
     672     assert(sim);
     673     bool end = true ;
     674     double dt = sim->simulate(eventID, end) ; // (QSim)
     675     return dt ;
     676 }




::

    N[blyth@localhost opticks]$ git log -n2
    commit 1761e9e4b69c3fd85eea7be8892dc59d1cdea255 (HEAD -> master, origin/master, origin/HEAD)
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Mon Jan 22 13:42:59 2024 +0800

        implement running from a sequence of input gensteps such that cxs_min_igs.sh can redo the pure Opticks GPU optical propagation for gensteps persisted from a prior Geant4+Opticks eg okjob/jok-tds job

    commit 507af61007daec200c3f0a912490950f3c910fba
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Mon Jan 22 12:08:46 2024 +0800

        add NPFold::set_allowempty_r to address opticks/notes/issues/avoiding_NPFold_save_of_empties_has_consequences_for_Galactic_material_with_no_props.rst used from U4Material::MakePropertyFold
    N[blyth@localhost opticks]$ 






smonitor.sh run of okjob.sh shows 0.003 GB/s leak
----------------------------------------------------

Workstation::

    GDB=1 ~/j/okjob.sh 
    ~/o/sysrap/smonitor.sh 

Laptop::

    ~/o/sysrap/smonitor.sh grab
    ~/o/sysrap/smonitor.sh ana


Getting okjob.sh going on N
-----------------------------

* had to rename /hpcfs to /old_hpcfs
* getting scrubbing of terminal output by somthing running after the primary job (sreport perhaps?)
* hit handling SEGV at end of job 
* adhoc leak check with "nvidia-smi -lms 1000"    does show leak : but arduous (3min init, and have to watch 
  as terminal output getting scrubbed
 
::

    GDB=1 ~/j/okjob.sh   ## delays the scrubbing 


::

    egin of Event --> 116
    2024-01-19 15:32:27.108 INFO  [306385] [QSim::simulate@376]  eventID 116 dt    0.009264 ph       9204 ph/M          0 ht       1748 ht/M          0 reset_ NO 
    2024-01-19 15:32:27.133 INFO  [306385] [SEvt::save@3953] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0/A116 [genstep,hit]
    junoSD_PMT_v2::EndOfEvent eventID 116 opticksMode 1 hitCollection 1748 hcMuon 0 GPU YES
    hitCollectionTT.size: 0	userhitCollectionTT.size: 0
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split begin. 2024-01-19 07:32:27.134933000Z
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split end. 2024-01-19 07:32:27.137078000Z
    junotoptask:DetSimAlg.execute   INFO: DetSimAlg Simulate An Event (117) 
    junoSD_PMT_v2::Initialize eventID 117
    Begin of Event --> 117
    2024-01-19 15:32:27.148 INFO  [306385] [QSim::simulate@376]  eventID 117 dt    0.009222 ph       8753 ph/M          0 ht       1673 ht/M          0 reset_ NO 
    2024-01-19 15:32:27.172 INFO  [306385] [SEvt::save@3953] /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0/A117 [genstep,hit]
    junoSD_PMT_v2::EndOfEvent eventID 117 opticksMode 1 hitCollection 1673 hcMuon 0 GPU YES
    hitCollectionTT.size: 0	userhitCollectionTT.size: 0
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split begin. 2024-01-19 07:32:27.173474000Z

    Thread 1 "python" received signal SIGSEGV, Segmentation fault.
    0x00007fffc8288da5 in DataModelWriterWithSplit::fill_hits(JM::SimEvt*, G4Event const*) () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/junosw/InstallArea/lib64/libAnalysisCode.so
    (gdb) 


    #0  0x00007fffc8288da5 in DataModelWriterWithSplit::fill_hits(JM::SimEvt*, G4Event const*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/junosw/InstallArea/lib64/libAnalysisCode.so
    #1  0x00007fffc828abf9 in DataModelWriterWithSplit::EndOfEventAction(G4Event const*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/junosw/InstallArea/lib64/libAnalysisCode.so
    #2  0x00007fffc7f27558 in MgrOfAnaElem::EndOfEventAction(G4Event const*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/junosw/InstallArea/lib64/libDetSimAlg.so
    #3  0x00007fffd1164242 in G4EventManager::DoProcessing(G4Event*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #4  0x00007fffc8403630 in G4SvcRunManager::SimulateEvent(int) () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/junosw/InstallArea/lib64/libG4SvcLib.so
    #5  0x00007fffc7f1d63a in DetSimAlg::execute() () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/junosw/InstallArea/lib64/libDetSimAlg.so
    #6  0x00007fffd4e3e511 in Task::execute() () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/sniper/InstallArea/lib64/libSniperKernel.so
    #7  0x00007fffd4e42c1d in TaskWatchDog::run() () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/sniper/InstallArea/lib64/libSniperKernel.so
    #8  0x00007fffd4e3e0b4 in Task::run() () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/sniper/InstallArea/lib64/libSniperKernel.so
    #9  0x00007fffd4ef8943 in boost::python::objects::caller_py_function_impl<boost::python::detail::caller<bool (Task::*)(), boost::python::default_call_policies, boost::mpl::vector2<bool, Task&> > >::operator()(_object*, _object*) () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/sniper/InstallArea/python/Sniper/libSniperPython.so
    #10 0x00007fffd4de65d5 in boost::python::objects::function::call(_object*, _object*) const ()


    
HUH, typing "bt" caused the scrubbing too. Some TERM messup ?   
But when the error is avoided by switching off edm get no scrubbing. 




Thrust Memory Management
--------------------------

* https://stackoverflow.com/questions/59265053/using-thrust-functions-with-raw-pointers-controlling-the-allocation-of-memory

 Checking code : i see no obvious mistakes. 



okjob.sh : terminal output is getting scrubbed
------------------------------------------------

::

      45608 sid    32396
      45609 sid    32397
      45610 sid    32398
      45611 sid    32399
    ]]stree::postcreate
    sdevice::Load failed read from  dirpath_ /hpcfs/juno/junogpu/blyth/.opticks/scontext dirpath /hpcfs/juno/junogpu/blyth/.opticks/scontext path /hpcfs/juno/junogpu/blyth/.opticks/scontext/sdevice.bin
    sdevice::Load failed read from  dirpath_ /hpcfs/juno/junogpu/blyth/.opticks/scontext dirpath /hpcfs/juno/junogpu/blyth/.opticks/scontext path /hpcfs/juno/junogpu/blyth/.opticks/scontext/sdevice.bin
    2024-01-19 15:06:29.294 FATAL [226832] [QRng::Load@79]  unabled to open file [/hpcfs/juno/junogpu/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin]
    2024-01-19 15:06:29.294 ERROR [226832] [QRng::Load@80] 
    QRng::Load_FAIL_NOTES
    =======================

    QRng::Load failed to load the curandState files. 
    These files should to created during *opticks-full* installation 
    by the bash function *opticks-prepare-installation* 
    which runs *qudarap-prepare-installation*. 

    Investigate by looking at the contents of the curandState directory, 
    as shown below::

        epsilon:~ blyth$ ls -l  ~/.opticks/rngcache/RNG/
        total 892336
        -rw-r--r--  1 blyth  staff   44000000 Oct  6 19:43 QCurandState_1000000_0_0.bin
        -rw-r--r--  1 blyth  staff  132000000 Oct  6 19:53 QCurandState_3000000_0_0.bin
        epsilon:~ blyth$ 



    python: /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/opticks/qudarap/QRng.cc:81: static curandState* QRng::Load(long int&, const char*): Assertion `!failed' failed.
     *** Break *** abort




QEventTest::setGenstep_many : NOT LEAKING
-------------------------------------------

Simple check shows no leak, staying at 653MiB throughout 

1. ~/o/qudarap/tests/QEventTest.sh
2. nvidia-smi -lms 500    # every half second  


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



