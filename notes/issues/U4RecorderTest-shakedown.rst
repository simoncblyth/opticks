U4RecorderTest-shakedown
===========================

What Next ?
-------------

* input random numbers
* basic ab.py like comparisons of SEvt
* gx level with geometry translation starting from Geant4
  rather than adhoc use of same geometry 



TODO : bring OpticksRandom over into U4
-----------------------------------------


TODO : debug input photon running with CXRaindropTest 
-------------------------------------------------------

Q: Why does QSimTest work but CXRaindropTest fail ?
A: The reason QSimTest works is that the adding of the genstep happens after QEvent is instanciated
    because it is not using the automated input photons which calls addGenstep very early, 
    but rather it manually sets input photons in mock_propagate which is after QSim/QEvent is instanciated. 


Try avoiding QEvent SEvt instanciation order sensitivity by deferring the SEvt::hostside_running_resize until the last 
possible moment SEvt::beginPhoton which is never called for deviceside running. 


::

    451 void QSimTest::mock_propagate()
    452 {
    453     assert( QSimLaunch::IsMutate(type)==true );
    454     LOG(info) << "[" ;
    455     LOG(info) << " SEventConfig::Desc " << SEventConfig::Desc() ;
    456 
    457     NP* p   = qs.duplicate_dbg_ephoton(num);
    458 
    459     SEvt::Get()->setInputPhoton(p);  // also adds placeholder genstep 
    460 





SMOKING GUN, the SEvt should not be self provider when using QEvent::

    2022-06-14 03:38:25.978 INFO  [327404] [SEvt::LoadInputPhoton@122]  SEventConfig::InputPhoton RandomSpherical10.npy path /home/blyth/.opticks/InputPhotons/RandomSpherical10.npy a.sstr (10, 4, 4, )
    2022-06-14 03:38:25.978 INFO  [327404] [SEvt::addGenstep@286]  s.desc sgs: idx   0 pho    10 off      0 typ INPUT_PHOTON gidx 0 enabled 1 tot_photon 10
    2022-06-14 03:38:25.978 INFO  [327404] [SEvt::setNumPhoton@315]  numphoton 10
    2022-06-14 03:38:25.978 INFO  [327404] [SEvt::resize@349]  is_self_provider 1
    sevent::descMax  evt.max_genstep 1000000 evt.max_photon  3000000 evt.max_simtrace  3000000 evt.max_bounce      9 evt.max_record     10 evt.max_rec      0
        sevent::descBuf 
        evt.genstep         N                    0        num_genstep       0        max_genstep 1000000
             evt.seed       N                    0           num_seed       0         max_photon 3000000
             evt.photon     Y            0x233f8f0         num_photon      10         max_photon 3000000
             evt.record     Y            0x233fb80         num_record     100         max_record      10
                evt.rec     N                    0            num_rec       0            max_rec       0
                evt.seq     N                    0            num_seq       0            max_seq       0
                evt.hit     N                    0            num_hit       0         max_photon 3000000
           evt.simtrace     N                    0       num_simtrace       0       max_simtrace 3000000

     sevent::descNum   evt.num_genstep     0 evt.num_seed     0 evt.num_photon    10 evt.num_simtrace     0 evt.num_record   100

    ...
    2022-06-14 03:38:28.379 INFO  [327404] [QEvent::setGenstep@160]  device_alloc genstep and seed 
    2022-06-14 03:38:28.380 INFO  [327404] [QEvent::setGenstep@165] SGenstep::DescGensteps gs.shape[0] 1 (10 ) total 10
    2022-06-14 03:38:28.380 ERROR [327404] [QEvent::setNumPhoton@577]  evt.photon is not nullptr : evt.photon : 0x233f8f0
    2022-06-14 03:38:28.380 INFO  [327404] [QEvent::uploadEvt@627] 
    sevent::descMax  evt.max_genstep 1000000 evt.max_photon  3000000 evt.max_simtrace  3000000 evt.max_bounce      9 evt.max_record     10 evt.max_rec      0
        sevent::descBuf 
        evt.genstep         Y       0x7f50cc000000        num_genstep       1        max_genstep 1000000
             evt.seed       Y       0x7f50fc600000           num_seed      10         max_photon 3000000
             evt.photon     Y            0x233f8f0         num_photon      10         max_photon 3000000
             evt.record     Y            0x233fb80         num_record     100         max_record      10
                evt.rec     N                    0            num_rec       0            max_rec       0
                evt.seq     N                    0            num_seq       0            max_seq       0
                evt.hit     N                    0            num_hit       0         max_photon 3000000
           evt.simtrace     N                    0       num_simtrace       0       max_simtrace 3000000

     sevent::descNum   evt.num_genstep     1 evt.num_seed    10 evt.num_photon    10 evt.num_simtrace     0 evt.num_record   100

    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (cudaMemcpy(reinterpret_cast<void*>( d ), h , size, cudaMemcpyHostToDevice ) ) failed with error: 'invalid argument' (/data/blyth/junotop/opticks/qudarap/QU.cc:344)

    ./cxs_raindrop.sh: line 111: 327404 Aborted                 (core dumped) $bin




::

    2022-06-14 02:46:59.099 INFO  [294307] [SBT::createGeom@109] ]
    2022-06-14 02:46:59.099 INFO  [294307] [SBT::getAS@584]  spec i0 c i idx 0
    2022-06-14 02:46:59.099 INFO  [294307] [QEvent::setGenstep@160]  device_alloc genstep and seed 
    2022-06-14 02:46:59.101 ERROR [294307] [QEvent::setNumPhoton@578]  evt.photon is not nullptr 
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (cudaMemcpy(reinterpret_cast<void*>( d ), h , size, cudaMemcpyHostToDevice ) ) failed with error: 'invalid argument' (/data/blyth/junotop/opticks/qudarap/QU.cc:344)


    Program received signal SIGABRT, Aborted.
    0x00007ffff3969387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libicu-50.2-4.el7_7.x86_64 libselinux-2.5-14.1.el7.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff3969387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff396aa78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff42a6cb3 in __gnu_cxx::__verbose_terminate_handler ()
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007ffff42ace26 in __cxxabiv1::__terminate(void (*)()) ()
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/eh_terminate.cc:47
    #4  0x00007ffff42ace61 in std::terminate () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/eh_terminate.cc:57
    #5  0x00007ffff42ad094 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff5d12440 <typeinfo for QUDA_Exception>, dest=
        0x7ffff5927a0a <QUDA_Exception::~QUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff59440bd in QU::copy_host_to_device<sphoton> (d=0x6c64e0, h=0x6c8730, num_items=10) at /data/blyth/junotop/opticks/qudarap/QU.cc:344
    #7  0x00007ffff591f4db in QEvent::setInputPhoton (this=0xf182f0) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:220
    #8  0x00007ffff591f230 in QEvent::setGenstep (this=0xf182f0, gs_=0x1a3c680) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:188
    #9  0x00007ffff591ed7c in QEvent::setGenstep (this=0xf182f0) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:149
    #10 0x00007ffff590feb0 in QSim::simulate (this=0xf1bbf0) at /data/blyth/junotop/opticks/qudarap/QSim.cc:234
    #11 0x000000000040d0b4 in main (argc=1, argv=0x7fffffff6458) at /data/blyth/junotop/opticks/CSGOptiX/tests/CXRaindropTest.cc:53
    (gdb) 


    (gdb) f 10
    #10 0x00007ffff590feb0 in QSim::simulate (this=0xf1bbf0) at /data/blyth/junotop/opticks/qudarap/QSim.cc:234
    234	   int rc = event->setGenstep(); 
    (gdb) f 9
    #9  0x00007ffff591ed7c in QEvent::setGenstep (this=0xf182f0) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:149
    149	    return gs == nullptr ? -1 : setGenstep(gs) ; 
    (gdb) f 8
    #8  0x00007ffff591f230 in QEvent::setGenstep (this=0xf182f0, gs_=0x1a3c680) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:188
    188	        setInputPhoton(); 
    (gdb) f 7
    #7  0x00007ffff591f4db in QEvent::setInputPhoton (this=0xf182f0) at /data/blyth/junotop/opticks/qudarap/QEvent.cc:220
    220	    QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)input_photon->bytes(), num_photon ); 
    (gdb) p evt->photon
    $1 = (sphoton *) 0x6c64e0
    (gdb) p input_photon
    $2 = (NP *) 0x6c6040
    (gdb) 


Looks like evt->photon address is on CPU, not on GPU as it should be. 
This is because it looks so similar to input_photons address which is highly unlikely for 
two pointers from two different address spaces. 



DONE : input photons
-----------------------

* input photons in both contexts : U4RecorderTest + CXRaindropTest

  * input NP array in common at SEvt level used from both contexts 

    * SEvt::SetInputPhotons rather than SEvt::AddTorchGensteps

  * usage level needs different treatment 

    1. qsim: uploading photons and getting qsim::generate_photon to use them 
 
       * DID this using placeholder input photon genstep
       * branch to handle input photon done in QEvent::setGenstep
         which invokes private method QEvent::setInputPhoton 

    2. U4Recorder needs to GeneratePrimaries using the input photon NP array  

       * input photon branch in SGenerate::GeneratePhotons that is called from U4VPrimaryGenerator::GeneratePrimaries
        

cx/CSGOptiX7.cu::

    192 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    193 {
    194     sevent* evt      = params.evt ;
    195     if (launch_idx.x >= evt->num_photon) return;
    196 
    197     unsigned idx = launch_idx.x ;  // aka photon_id
    198     unsigned genstep_id = evt->seed[idx] ;
    199     const quad6& gs     = evt->genstep[genstep_id] ;
    200 
    201     qsim* sim = params.sim ;
    202     curandState rng = sim->rngstate[idx] ;    // TODO: skipahead using an event_id 
    203 
    204     sphoton p = {} ;
    205 
    206     sim->generate_photon(p, rng, gs, idx, genstep_id );
    207 

* seeding relates a photon slot with its corresponding genstep, just requiring the genstep 
  to have the photon count 
* better not to change the pattern just for input photons, even though input photon running 
  has no need for seeding (or gensteps for that matter).  
* to keep the pattern use placeholder "input photon gensteps"

::

    1351 inline QSIM_METHOD void qsim::generate_photon(sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    1352 {
    1353     const int& gencode = gs.q0.i.x ;
    1354 
    1355     switch(gencode)
    1356     {
    1357         case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ;
    1358         case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ;
    1359         case OpticksGenstep_CERENKOV:        cerenkov->generate(     p, rng, gs, photon_id, genstep_id ) ; break ;
    1360         case OpticksGenstep_SCINTILLATION:   scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ;
    1361         default:                             generate_photon_dummy(  p, rng, gs, photon_id, genstep_id)  ; break ;
    1362     }
    1363 }

    
* DONE: Added OpticksGenstep_INPUT_PHOTON 

::

    0231 double QSim::simulate()
     232 {
     233    int rc = event->setGenstep();
     234    double dt = rc == 0 && cx != nullptr ? cx->simulate() : -1. ;
     235    return dt ;
     236 }

    143 int QEvent::setGenstep()
    144 {
    145     NP* gs = SEvt::GetGenstep();
    146     SEvt::Clear();   // clear the quad6 vector, ready to collect more genstep
    147     if(gs == nullptr) LOG(fatal) << "Must SEvt::AddGenstep before calling QEvent::setGenstep " ;
    148     return gs == nullptr ? -1 : setGenstep(gs) ;
    149 }

    151 int QEvent::setGenstep(NP* gs_)
    152 {
    153     gs = gs_ ;
    154     SGenstep::Check(gs);
    155     evt->num_genstep = gs->shape[0] ;
    156 
    157     if( evt->genstep == nullptr && evt->seed == nullptr )
    158     {
    159         LOG(info) << " device_alloc genstep and seed " ;
    160         evt->genstep = QU::device_alloc<quad6>( evt->max_genstep ) ;
    161         evt->seed    = QU::device_alloc<int>(   evt->max_photon )  ;
    162     }
    163 
    164     LOG(LEVEL) << SGenstep::Desc(gs, 10) ;
    165 
    166     bool num_gs_allowed = evt->num_genstep <= evt->max_genstep ;
    167     if(!num_gs_allowed) LOG(fatal) << " evt.num_genstep " << evt->num_genstep << " evt.max_genstep " << evt->max_genstep ;
    168     assert( num_gs_allowed );
    169 
    170     QU::copy_host_to_device<quad6>( evt->genstep, (quad6*)gs->bytes(), evt->num_genstep );
    171 
    172     QU::device_memset<int>(   evt->seed,    0, evt->max_photon );
    173 
    174     //count_genstep_photons();   // sets evt->num_seed
    175     //fill_seed_buffer() ;       // populates seed buffer
    176     count_genstep_photons_and_fill_seed_buffer();   // combi-function doing what both the above do 
    177 
    178 
    179     int gencode0 = SGenstep::GetGencode(gs, 0); // gencode of first genstep   
    180 
    181     if(OpticksGenstep_::IsFrame(gencode0))
    182     {
    183         setNumSimtrace( evt->num_seed );
    184     }
    185     else
    186     {
    187         setNumPhoton( evt->num_seed );  // photon, rec, record may be allocated here depending on SEventConfig
    188     }


* HMM: in spirit of not breaking the pattern for input photons, calling SEvt::SetInputPhotons(NP*) 
  needs to Add INPUT_PHOTON genstep : then the above can proceed unchanged for input photons


::

    258 /**
    259 QEvent::setPhoton
    260 -------------------
    261 
    262 This is only used with non-standard input photon running, 
    263 eg the photon mutatating QSimTest use this.  
    264 The normal mode of operation is to start from gensteps using QEvent::setGenstep
    265 and seed and generate photons on device.
    266 
    267 HMM: this is problematic as it breaks the pattern of normal genstep running 
    268 
    269 **/
    270 
    271 void QEvent::setPhoton(const NP* p_)
    272 {
    273     p = p_ ;
    274     
    275     int num_photon = p->shape[0] ;
    276     
    277     LOG(info) << "[ " <<  p->sstr() << " num_photon " << num_photon  ;
    278     
    279     assert( p->has_shape( -1, 4, 4) );
    280     
    281     setNumPhoton( num_photon );
    282     
    283     QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)p->bytes(), num_photon );
    284     
    285     LOG(info) << "] " <<  p->sstr() << " num_photon " << num_photon  ;
    286 }   





input photon mock_propagate getNumHit assert : ASSUMED TO BE COLLATERAL DAMAGE FROM PRD SIZE INCONSISTENCY
------------------------------------------------------------------------------------------------------------


::

    0  407 	    assert( evt->photon ); 
       408 	    assert( evt->num_photon ); 
       409 	
    -> 410 	    evt->num_hit = SU::count_if_sphoton( evt->photon, evt->num_photon, *selector );    
       411 	
       412 	    LOG(info) << " evt.photon " << evt->photon << " evt.num_photon " << evt->num_photon << " evt.num_hit " << evt->num_hit ;  
       413 	    return evt->num_hit ; 
    (lldb) p evt->photon
    (sphoton *) $0 = 0x000000070a240000
    (lldb) p evt->num_photon
    (int) $1 = 8
    (lldb) f 11
    frame #11: 0x0000000100646ecc libSysRap.dylib`SU::count_if_sphoton(sphoton const*, unsigned int, sphoton_selector const&) + 44
    libSysRap.dylib`SU::count_if_sphoton:
        0x100646ecc <+44>: addq   $0x10, %rsp
        0x100646ed0 <+48>: popq   %rbp
        0x100646ed1 <+49>: retq   
        0x100646ed2 <+50>: nopw   %cs:(%rax,%rax)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff72d94b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff72f5f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff72cf01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff70beaf8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff70beb113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff7202ceab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff70c067c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff70c0626f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x000000010064a5b6 libSysRap.dylib`void thrust::cuda_cub::free<thrust::cuda_cub::tag, thrust::pointer<long, thrust::cuda_cub::tag, thrust::use_default, thrust::use_default> >(thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag>&, thrust::pointer<long, thrust::cuda_cub::tag, thrust::use_default, thrust::use_default>) + 166
        frame #9: 0x0000000100649508 libSysRap.dylib`thrust::detail::temporary_allocator<long, thrust::cuda_cub::tag>::allocate(unsigned long) + 72
        frame #10: 0x000000010064c9c3 libSysRap.dylib`long thrust::cuda_cub::reduce_n<thrust::cuda_cub::tag, thrust::cuda_cub::transform_input_iterator_t<long, thrust::device_ptr<sphoton const>, sphoton_selector>, long, long, thrust::plus<long> >(thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag>&, thrust::cuda_cub::transform_input_iterator_t<long, thrust::device_ptr<sphoton const>, sphoton_selector>, long, long, thrust::plus<long>) + 67
      * frame #11: 0x0000000100646ecc libSysRap.dylib`SU::count_if_sphoton(sphoton const*, unsigned int, sphoton_selector const&) + 44
        frame #12: 0x00000001001acd01 libQUDARap.dylib`QEvent::getNumHit(this=0x0000000100991d10) const at QEvent.cc:410
        frame #13: 0x000000010001a606 QSimTest`QSimTest::mock_propagate(this=0x00007ffeefbfe3c8) at QSimTest.cc:457
        frame #14: 0x000000010001c379 QSimTest`QSimTest::main(this=0x00007ffeefbfe3c8) at QSimTest.cc:634
        frame #15: 0x000000010001d24b QSimTest`main(argc=1, argv=0x00007ffeefbfe6a8) at QSimTest.cc:659
        frame #16: 0x00007fff72c44015 libdyld.dylib`start + 1
        frame #17: 0x00007fff72c44015 libdyld.dylib`start + 1
    (lldb) f 13
    frame #13: 0x000000010001a606 QSimTest`QSimTest::mock_propagate(this=0x00007ffeefbfe3c8) at QSimTest.cc:457
       454 	    qs.mock_propagate( prd, type ); 
       455 	
       456 	    const QEvent* event = qs.event ; 
    -> 457 	    unsigned num_hit = event->getNumHit(); 
       458 	    LOG(info) << " num_hit " << num_hit ;
       459 	
       460 	    SEvt::Save(dir); 
    (lldb) 



After commenting the above QSimTest getNumHit find the standard SEvt getHit succeeds::

    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    2022-06-13 13:14:23.314 INFO  [22054730] [QSim::mock_propagate@823] ]
    2022-06-13 13:14:23.314 INFO  [22054730] [SEvt::save@847]  dir /tmp/blyth/opticks/QSimTest/mock_propagate
    2022-06-13 13:14:23.314 FATAL [22054730] [QEvent::getPhoton@320] [ evt.num_photon 8 p.sstr (8, 4, 4, ) evt.photon 0x70a240000
    2022-06-13 13:14:23.314 FATAL [22054730] [QEvent::getPhoton@323] ] evt.num_photon 8
    2022-06-13 13:14:23.314 FATAL [22054730] [*QEvent::getRecord@374]  getRecord called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-13 13:14:23.314 FATAL [22054730] [*QEvent::getRec@386]  getRec called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-13 13:14:23.314 FATAL [22054730] [*QEvent::getSeq@363]  getSeq called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-13 13:14:23.316 INFO  [22054730] [*QEvent::getHit@454]  evt.photon 0x70a240000 evt.num_photon 8 evt.num_hit 4 selector.hitmask 64 SEventConfig::HitMask 64 SEventConfig::HitMaskLabel SD
    2022-06-13 13:14:23.316 INFO  [22054730] [*QEvent::getHit_@481]  hit.sstr (4, 4, 4, )
    2022-06-13 13:14:23.316 FATAL [22054730] [*QEvent::getSimtrace@345]  getSimtrace called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-13 13:14:23.316 INFO  [22054730] [SEvt::save@851] SEvt::descComponent
     SEventConfig::CompMaskLabel genstep,photon,record,rec,seq,seed,hit,simtrace,domain
                     hit          (4, 4, 4, ) 
                    seed                (8, ) 
                 genstep          (1, 6, 4, )       SEventConfig::MaxGenstep                   0


Is there a problem with calling getNumHit twice ?


Is s.optical being filled ?::

    //_QSim_mock_propagate idx 7 evt.num_photon 8 evt.max_record 0  
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.propagate idx 0 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.propagate idx 1 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.propagate idx 2 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.propagate idx 3 bounce 0 command 3 flag 0 s.optical.x 2 
    //qsim.propagate idx 4 bounce 0 command 3 flag 0 s.optical.x 716983765 
    //qsim.propagate idx 5 bounce 0 command 3 flag 0 s.optical.x -268435473 
    //qsim.propagate idx 6 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.propagate idx 7 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.mock_propagate idx 0 bounce 1 evt.max_bounce 9 command 2 
    //qsim.mock_propagate idx 1 bounce 1 evt.max_bounce 9 command 2 
    //qsim.mock_propagate idx 2 bounce 1 evt.max_bounce 9 command 2 



Non-sensical prd from idx 4::

    //qsim.mock_propagate evt.max_bounce 9 evt.max_record 0 evt.record 0x0 evt.num_record 0 evt.num_rec 0 
    //qsim.mock_propagate idx 0 prd.q0.f.xyzw (    0.0000     0.0000     1.0000   100.0000) 
    //qsim.mock_propagate idx 1 prd.q0.f.xyzw (    0.0000     0.0000     1.0000   200.0000) 
    //qsim.mock_propagate idx 2 prd.q0.f.xyzw (    0.0000     0.0000     1.0000   300.0000) 
    //qsim.mock_propagate idx 3 prd.q0.f.xyzw (    0.0000     0.0000     1.0000   400.0000) 
    //qsim.mock_propagate idx 4 prd.q0.f.xyzw (       nan -2658455674657181688750263746384887808.0000    -2.0000        nan) 
    //qsim.mock_propagate idx 5 prd.q0.f.xyzw (       nan        nan        nan        nan) 
    //qsim.mock_propagate idx 6 prd.q0.f.xyzw (    0.0000     0.0000     0.0000     0.0000) 
    //qsim.mock_propagate idx 7 prd.q0.f.xyzw (    0.0000     0.0000     0.0000     0.0000) 

* FIXED THIS : IT WAS DUE TO SEventConfig inconsistency in QSimTest initializtion, 
  had to change order of instanciation 

Huh looks like prd using a different max_bounce to propagation::

      : t.prd                                              :         (8, 4, 2, 4) : 0:01:21.105138 


FIXED : Discrepant max bounce::

    epsilon:tests blyth$ grep SetMaxBounce *.*
    QSimTest.cc:        SEventConfig::SetMaxBounce(num_bounce); 


::

    In [2]: t.prd                                                                                                                                                               
    Out[2]: 
    array([[[[  0.,   0.,   1., 100.],
             [  0.,   0.,   0.,   0.]],

            [[  0.,   0.,   1., 200.],
             [  0.,   0.,   0.,   0.]],

            [[  0.,   0.,   1., 300.],
             [  0.,   0.,   0.,   0.]],

            [[  0.,   0.,   1., 400.],
             [  0.,   0.,   0.,   0.]]],


           [[[  0.,   0.,   1., 100.],
             [  0.,   0.,   0.,   0.]],

            [[  0.,   0.,   1., 200.],
             [  0.,   0.,   0.,   0.]],

            [[  0.,   0.,   1., 300.],
             [  0.,   0.,   0.,   0.]],

            [[  0.,   0.,   1., 400.],
             [  0.,   0.,   0.,   0.]]],




DONE : More featureful geometry, in u4/tests/U4RecorderTest.cc GEOM RaindropRockAirWater
------------------------------------------------------------------------------------------

* need more featureful geometry to test/develop things like microstep skipping 

  * before full geometry prep a local simple Raindrop geometry 
  * need water and air 



Geant4 originals : expand from just LS_ori to all materials 
--------------------------------------------------------------

::

    0805 void GPropertyLib::addRawOriginal(GPropertyMap<double>* pmap)
     806 {
     807     m_raw_original.push_back(pmap);
     808 }
     ...
     845 GPropertyMap<double>* GPropertyLib::getRawOriginal(const char* shortname) const
     846 {
     847     unsigned num_raw_original = m_raw_original.size();
     848     for(unsigned i=0 ; i < num_raw_original ; i++)
     849     { 
     850         GPropertyMap<double>* pmap = m_raw_original[i];
     851         const char* name = pmap->getShortName();
     852         if(strcmp(shortname, name) == 0) return pmap ;
     853     }
     854     return NULL ;
     855 }

    epsilon:ggeo blyth$ opticks-f addRawOriginal
    ./extg4/X4PhysicalVolume.cc:        m_sclib->addRawOriginal(pmap);      
    ./extg4/X4MaterialTable.cc:        m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib
    ./ggeo/GPropertyLib.cc:void GPropertyLib::addRawOriginal(GPropertyMap<double>* pmap)
    ./ggeo/GPropertyLib.hh:        void                  addRawOriginal(GPropertyMap<double>* pmap);
    epsilon:opticks blyth$ 


     342 void X4PhysicalVolume::collectScintillatorMaterials()
     343 {
     ...
     348     typedef GPropertyMap<double> PMAP ;
     349     std::vector<PMAP*> raw_energy_pmaps ;
     350     m_mlib->findRawOriginalMapsWithProperties( raw_energy_pmaps, SCINTILLATOR_PROPERTIES, ',' );
     ...
     378     // original energy domain 
     379     for(unsigned i=0 ; i < num_scint ; i++)
     380     {
     381         PMAP* pmap = raw_energy_pmaps[i] ;
     382         m_sclib->addRawOriginal(pmap);
     383     }

    105 void X4MaterialTable::init()
    106 {
    107     unsigned num_input_materials = m_input_materials.size() ;
    ...
    111     for(unsigned i=0 ; i < num_input_materials ; i++)
    112     {
    ...
    136         char mode_asis_en = 'E' ;
    137         GMaterial* rawmat_en = X4Material::Convert( material, mode_asis_en );
    138         GPropertyMap<double>* pmap_rawmat_en = dynamic_cast<GPropertyMap<double>*>(rawmat_en) ;
    139         m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib

    0887 void GPropertyLib::findRawOriginalMapsWithProperties( std::vector<GPropertyMap<double>*>& dst, const char* props, char delim )
     888 {
     889     SelectPropertyMapsWithProperties(dst, props, delim, m_raw_original );
     890 }

    0982 void GPropertyLib::saveRawOriginal()
     983 {
     984     std::string dir = getCacheDir();
     985     unsigned num_raw_original = m_raw_original.size();
     986     LOG(LEVEL) << "[ " << dir << " num_raw_original " << num_raw_original ;
     987     for(unsigned i=0 ; i < num_raw_original ; i++)
     988     {
     989         GPropertyMap<double>* pmap = m_raw_original[i] ;
     990         pmap->save(dir.c_str());
     991     }
     992     LOG(LEVEL) << "]" ;
     993 }

    001 #include "SConstant.hh"
      2 
      3 const char* SConstant::ORIGINAL_DOMAIN_SUFFIX = "_ori" ;
      4 

    1076 template <typename T>
    1077 void GPropertyMap<T>::save(const char* dir)
    1078 {
    1079     std::string shortname = m_shortname ;
    1080     if(m_original_domain) shortname += SConstant::ORIGINAL_DOMAIN_SUFFIX ;
    1081 
    1082     LOG(LEVEL) << " save shortname (+_ori?) [" << shortname << "] m_original_domain " << m_original_domain  ;
    1083 
    1084     for(std::vector<std::string>::iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
    1085     {
    1086         std::string key = *it ;
    1087         std::string propname(key) ;
    1088         propname += ".npy" ;
    1089 
    1090         GProperty<T>* prop = m_prop[key] ;
    1091         prop->save(dir, shortname.c_str(), propname.c_str());  // dir, reldir, name
    1092     }
    1093 }


geocache-create uses okg4/tests/OKX4Test.cc::

    112     
    113     m_ggeo->postDirectTranslation();   // closing libs, finding repeat instances, merging meshes, saving 
    114     

    0584 /**
     585 GGeo::postDirectTranslation
     586 -------------------------------
     587 
     588 Invoked from G4Opticks::translateGeometry after the X4PhysicalVolume conversion
     589 for live running or from okg4/tests/OKX4Test.cc main for geocache-create.
     590 
     591 **/
     592 
     593 
     594 void GGeo::postDirectTranslation()
     595 {
     596     LOG(LEVEL) << "[" ;
     597 
     598     prepare();     // instances are formed here     
     599 
     600     LOG(LEVEL) << "( GBndLib::fillMaterialLineMap " ;
     601     GBndLib* blib = getBndLib();
     602     blib->fillMaterialLineMap();
     603     LOG(LEVEL) << ") GBndLib::fillMaterialLineMap " ;
     604 
     605     LOG(LEVEL) << "( GGeo::save " ;
     606     save();
     607     LOG(LEVEL) << ") GGeo::save " ;
     608 
     609 
     610     deferred();
     611 
     612     postDirectTranslationDump();
     613 
     614     LOG(LEVEL) << "]" ;
     615 }


With Gun : First 100 label id are zero ? FIXED 
------------------------------------------------

::

    In [25]: np.all( id_[100:] == np.arange(100,388, dtype=np.int32)  )
    Out[25]: True

    In [26]: np.all( id_[:100] == 0 )
    Out[26]: True

FIXED by commenting the SEvt::AddTorchGenstep when gun running::

    133 int main(int argc, char** argv)
    134 {    
    135     OPTICKS_LOG(argc, argv);
    136 
    137     unsigned max_bounce = 9 ;
    138     SEventConfig::SetMaxBounce(max_bounce);
    139     SEventConfig::SetMaxRecord(max_bounce+1);
    140     SEventConfig::SetMaxRec(max_bounce+1);
    141     SEventConfig::SetMaxSeq(max_bounce+1);
    142 
    143     SEvt evt ; 
    144     //SEvt::AddTorchGenstep();


With Gun : FIXED : Unexpected seq labels 
-----------------------------------------

* should be starting with SI or CK 

::

   0 : MI SD SD SD MI MI 
   1 : MI SD SD SD MI MI 
   2 : MI SD SD MI MI MI 
   3 : MI SD SD MI MI MI 
   4 : MI SC SD MI MI MI 
   5 : SI SC SD MI MI MI 
   6 : SI SC SD MI MI MI 
   7 : SI AB AB MI 
   8 : SI AB AB MI 


After zeroing seq and rec at SEvt::startPhoton the seq looks more reasonable::

   0 : CK AB AB 
   1 : CK AB SC AB MI 
   2 : CK AB 
   3 : CK MI 
   4 : CK AB 
   5 : SI AB 
   6 : SI SC MI MI MI MI 
   7 : SI AB 
   8 : SI AB AB MI 
   9 : SI MI 


With Gun : Not terminated at AB ? Probably reemision rejoin AB scrub not working yet ? YEP: FIXED
----------------------------------------------------------------------------------------------------

* actually did i implement that at all ? only did the flagmask not the seqhis ?

seqhis::

   0 : CK AB AB 
   1 : CK AB SC AB MI 
   2 : CK AB 
   3 : CK MI 
   4 : CK AB 
   5 : SI AB 
   6 : SI SC MI MI MI MI 
   7 : SI AB 
   8 : SI AB AB MI 
   9 : SI MI 

Implement GIDX control for debug running with single genstep.::

    bflagdesc_(r[0,j])
     idx(     0) prd(  0    0     0 0 ii:    0)  CK               CK  
     idx(     0) prd(  0    0     0 0 ii:    0)  AB            AB|CK  
     idx(     0) prd(  0    0     0 0 ii:    0)  AB         RE|AB|CK  


* FIXED : clear discrepancy between the flag+seqhis and the flagmask 

The current_photon flag gets seq.add_nibble by SEvt::pointPhoton::

    342 void SEvt::pointPhoton(const spho& label)
    343 {   
    344     assert( label.isSameLineage(current_pho) );
    345     unsigned idx = label.id ;
    346     int& bounce = slot[idx] ;
    347     
    348     const sphoton& p = current_photon ;
    349     srec& rec = current_rec ;
    350     sseq& seq = current_seq ;
    351     
    352     if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;
    353     if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );  
    354     if( evt->seq    && bounce < evt->max_seq    ) seq.add_nibble(bounce, p.flag(), p.boundary() );
    355     
    356     bounce += 1 ;
    357 }

Fixed reemission bookkeeping by history rewrite.

SEvt::rjoinPhoton::


    331     if( evt->photon )
    332     {
    333        // HMM: could directly change photon[idx] via ref ? 
    334        // But are here taking a copy to current_photon
    335        // and relying on copyback at SEvt::endPhoton
    336 
    337         current_photon = photon[idx] ;
    338         assert( current_photon.flag() == BULK_ABSORB );
    339         assert( current_photon.flagmask & BULK_ABSORB  );   // all continuePhoton should have BULK_ABSORB in flagmask
    340 
    341         current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
    342         current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask 
    343     }
    344 
    345     if( evt->seq )
    346     {
    347         current_seq = seq[idx] ;
    348         unsigned seq_flag = current_seq.get_flag(prior);
    349         assert( seq_flag == BULK_ABSORB );
    350         current_seq.set_flag(prior, BULK_REEMIT);
    351     }
    352 
    353     if( evt->record )
    354     {
    355         sphoton& rjoin_record = evt->record[evt->max_record*idx+prior]  ;
    356         unsigned rjoin_flag = rjoin_record.flag() ;
    357 
    358         LOG(info) << " rjoin.flag "  << OpticksPhoton::Flag(rjoin_flag)  ;
    359         assert( rjoin_flag == BULK_ABSORB );
    360         assert( rjoin_record.flagmask & BULK_ABSORB );
    361 
    362         rjoin_record.flagmask &= ~BULK_ABSORB ; // scrub BULK_ABSORB from flagmask  
    363         rjoin_record.set_flag(BULK_REEMIT) ;
    364     }


GIDX selection beyond the first is asserting : FIXED 
--------------------------------------------------------

::

    2022-06-09 16:52:41.855 INFO  [19428647] [U4Recorder::BeginOfRunAction@38] 
    2022-06-09 16:52:41.855 INFO  [19428647] [U4Recorder::BeginOfEventAction@40] 
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   0 pho     5 off      0 typ G4Cerenkov_modified gidx 0 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   0 pho     1 off      0 typ DsG4Scintillation_r4695 gidx 1 enabled 1
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::setNumPhoton@210]  numphoton 1
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 2 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 3 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 4 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::beginPhoton@269] 
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::beginPhoton@270] spho ( gs ix id gn   1   0    1 0 ) 
    2022-06-09 16:52:41.856 ERROR [19428647] [SEvt::beginPhoton@275]  not in_range  idx 1 pho.size  1 label spho ( gs ix id gn   1   0    1 0 ) 
    Assertion failed: (in_range), function beginPhoton, file /Users/blyth/opticks/sysrap/SEvt.cc, line 281.
    ./U4RecorderTest.sh: line 43: 73818 Abort trap: 6           U4RecorderTest
    === ./U4RecorderTest.sh : logdir /tmp/blyth/opticks/U4RecorderTest
    epsilon:tests blyth$ 



The sgs genstep labelling is using an offset that does not account for enabled gensteps presumably::

     56 inline spho sgs::MakePho(unsigned idx, const spho& ancestor)
     57 {
     58     return ancestor.isDefined() ? ancestor.make_reemit() : spho::MakePho(index, idx, offset + idx, 0) ;
     59 }


FIXED this by simplifying genstep disabling to simply set the numphotons of disabled gensteps to zero, 
without any change to the collection machinery.  As genstep disabling is purely for debugging this is acceptable. 




FIXED : Checking rjoinPhoton matching tripping some asserts
---------------------------------------------------------------


::

    u4 ; cd tests

    epsilon:tests blyth$ ./U4RecorderTest.sh


    2022-06-09 20:51:29.134 INFO  [19769941] [SEvt::rjoinPhoton@315] 
    2022-06-09 20:51:29.134 INFO  [19769941] [SEvt::rjoinPhoton@316] spho ( gs ix id gn 117   0  33310 ) 
    rjoinPhotonCheck : does not have BULK_ABSORB flag ? ph.idx 333 flag_AB NO flagmask_AB NO
     pos (-1000.000,722.148,670.385)  t  46.844
     mom (-0.814, 0.581,-0.026)  iindex 0
     pol (-0.145,-0.159, 0.977)  wl 394.830
     bn 0 fl 4 id 0 or 1.000 ix 333 fm 16 ab MI
     digest(16) 1bf2798f0385a6f99531161605e3e661
     digest(12) 62c0957fc9dbf3ed296559467aa5d5d5
     NOT seq_flag_AB, rather   
     rjoin_record_d12   1e80c7b62fe41f2b3cfbc743988d1787
     current_photon_d12 62c0957fc9dbf3ed296559467aa5d5d5
     d12_match NO
    Assertion failed: (d12_match), function rjoinPhoton, file /Users/blyth/opticks/sysrap/SEvt.cc, line 377.
    ./U4RecorderTest.sh: line 43: 23381 Abort trap: 6           U4RecorderTest
    === ./U4RecorderTest.sh : logdir /tmp/blyth/opticks/U4RecorderTest
    /Users/blyth/opticks/u4/tests
    cfbase:/usr/local/opticks/geocache/OKX4Test_lWorld0x5780b30_PV_g4live/g4ok_gltf/5303cd587554cb16682990189831ae83/1/CSG_GGeo 
    Fold : setting globals False globals_prefix  
    t



FIXED : Smoking gun is getting impossible rjoin.flag of SCINTILLATION are clearly 
wandering over to another photons records::

    2022-06-10 11:56:09.859 INFO  [19958285] [SEvt::rjoinPhoton@321] 
    2022-06-10 11:56:09.859 INFO  [19958285] [SEvt::rjoinPhoton@322] spho (gs:ix:id:gn 117   0    0 10)
    rjoinPhotonCheck : does not have BULK_ABSORB flag ? sphoton idx 0 flag MISS flagmask SI|MI|RE
     pos (-1000.000,722.148,670.385)  t  46.844
     mom (-0.814, 0.581,-0.026)  iindex 0
     pol (-0.145,-0.159, 0.977)  wl 394.830
     bn 0 fl 4 id 0 or 1.000 ix 0 fm 16 ab MI
     digest(16) 7706526a21ed79f8fb759805c75c798b
     digest(12) 62c0957fc9dbf3ed296559467aa5d5d5
     NOT seq_flag_AB, rather   
     idx 0 bounce 11 prior 10 evt.max_record 10 rjoin_record_d12   1e80c7b62fe41f2b3cfbc743988d1787
     current_photon_d12 62c0957fc9dbf3ed296559467aa5d5d5
     d12match NO
     rjoin_record 
     pos (-9.399,42.455,114.610)  t  7.007
     mom ( 0.802, 0.597, 0.017)  iindex 0
     pol ( 0.559,-0.739,-0.377)  wl 466.605
     bn 0 fl 2 id 0 or 1.000 ix 1 fm 2 ab SI
     digest(16) 07cb368115014bb1c643bd028d48c1e0
     digest(12) 1e80c7b62fe41f2b3cfbc743988d1787
    2022-06-10 11:56:09.860 INFO  [19958285] [SEvt::rjoinPhoton@400]  rjoin.flag SCINTILLATION
     NOT rjoin_flag_AB 
     NOT rjoin_record_flagmask_AB 
     current_photon 
     pos (-1000.000,722.148,670.385)  t  46.844
     mom (-0.814, 0.581,-0.026)  iindex 0
     pol (-0.145,-0.159, 0.977)  wl 394.830
     bn 0 fl 10 id 0 or 1.000 ix 0 fm 16 ab RE
     digest(16) 829c294403eff470277c9cdb81f983a6
     digest(12) 62c0957fc9dbf3ed296559467aa5d5d5
    2022-06-10 11:56:09.860 INFO  [19958285] [SEvt::pointPhoton@494] spho (gs:ix:id:gn 117   0    0 10)  seqhis      55555555552 nib 11 SI RE RE RE RE RE RE RE RE RE RE                
    2022-06-10 11:56:09.860 INFO  [19958285] [U4Recorder::UserSteppingAction_Optical@190]  step.tstat fStopAndKill MISS



Must review how evt->max_record truncation is handled, as apparently not working.

* FIXED : the problem was just with the rjoin checking not applying the truncation







