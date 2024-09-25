ilker-hot-spots
================

Report from Ilker : presentation on google docs shows low energy looking clumpy as if lower stats that g4 even with same stats
----------------------------------------------------------------------------------------------------------------------------------

* duplicated randoms/photons maybe ? 


::

    Hello Simon, 

    I hope you are doing well. I have been doing some comparisons with Opticks and
    G4. I have noticed that there are some artifacts and hot spots show up  on hits
    whenever photon yields are low, if I increase the yields , hits more look like
    G4. I wonder if you ever seen this behavior before? 
     
    I also noticed that the hit counts agree much better when i turn off the
    reflections on the steel. I introduce 20%  surface reflections on steel by
    using REFLECTIVITY property.
     
    Also i am curious if there is a way of checking boundary overlaps in Opticks
    and how could i do a good geometry comparison? I have been doing indirectly by
    looking at final position of photons and compare it both G4 and Opticks yet may
    be there is an alternative way you may know.
     
    Here is the presentation to the plots and some comparisons I have done.  If you
    have any suggestions , i really appreciate.
     
    Thank you,
    Ilker


    https://docs.google.com/presentation/d/1d-d0UzUmJtOr5QkehVI_V5LbrztAPJJCSDxZTfy9XFo/edit?pli=1#slide=id.g2f87c7b96b1_0_87



Missing skipahead causing recycling of randoms resulting in the clumping probably 
-----------------------------------------------------------------------------------

cx/CSGOptiX7.cu::

    274 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    275 {
    276     sevent* evt = params.evt ;
    277     if (launch_idx.x >= evt->num_photon) return;
    278 
    279     unsigned idx = launch_idx.x ;  // aka photon_idx
    280     unsigned genstep_idx = evt->seed[idx] ;
    281     const quad6& gs = evt->genstep[genstep_idx] ;
    282 
    283     qsim* sim = params.sim ;
    284     curandState rng = sim->rngstate[idx] ;    // TODO: skipahead using an event_id 
    285 

TRY cx/CSGOptiX7.cu::

    274 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    275 {
    276     sevent* evt = params.evt ;
    277     if (launch_idx.x >= evt->num_photon) return;
    278 
    279     unsigned idx = launch_idx.x ;  // aka photon_idx
    280     unsigned genstep_idx = evt->seed[idx] ;
    281     const quad6& gs = evt->genstep[genstep_idx] ;
    282 
    283     qsim* sim = params.sim ;
    284 
    285 #ifdef OLD_WITHOUT_SKIPAHEAD
    286     curandState rng = sim->rngstate[idx] ;
    287 #else
    288     curandState rng ;  
    289     //sim->rng->get_rngstate_with_skipahead( rng, sim->evt->index, idx );   // awkward because sim otherwise entirely set at initialization 
    290     sim->rng->get_rngstate_with_skipahead( rng, params.event_index, idx );
    291 #endif



How to get the event_idx GPU side ?
------------------------------------

* 


changes
----------

* add SEventConfig::EventSkipahead OPTICKS_EVENT_SKIPAHEAD
* use that from QSim::UploadComponents to pass into QRng ctor
* add qrng.h:rng to qsim.h 
* use qrng::get_rngstate_with_skipahead in CSGOptiX7.cu simulate
* get event_idx GPU side for skipahead offsetting via sevent.h with sim->evt->index



skipahead usage
----------------------

::

     18 /**
     19 QRng::QRng
     20 ------------
     21 
     22 Instanciation is invoked from QSim::UploadComponents
     23 
     24 **/
     25 
     26 QRng::QRng(unsigned skipahead_event_offset)
     27     :
     28     path(SCurandState::Path()),        // null path will assert in Load
     29     rngmax(0),
     30     rng_states(Load(rngmax, path)),   // rngmax set based on file_size/item_size 
     31     qr(new qrng(skipahead_event_offset)),
     32     d_qr(nullptr)
     33 {   
     34     INSTANCE = this ;
     35     upload(); 
     36     bool uploaded = d_qr != nullptr ; 
     37     LOG_IF(fatal, !uploaded) << " FAILED to upload curand states " ;
     38     assert(uploaded);
     39 }






     31 #if defined(__CUDACC__) || defined(__CUDABE__)
     32 
     33 #include <curand_kernel.h>
     34     
     35 /**
     36 qrng::random_setup
     37 ---------------------
     38 
     39 light touch encapsulation of setup only as want generation of randoms to be familiar/standard and suffer no overheads
     40 
     41 **/
     42 
     43 inline QRNG_METHOD void qrng::random_setup(curandStateXORWOW& rng, unsigned event_idx, unsigned photon_idx )
     44 {
     45     unsigned long long skipahead_ = skipahead_event_offset*event_idx ;
     46     rng = *(rng_states + photon_idx) ; 
     47     skipahead( skipahead_, &rng ); 
     48 }
     49 #endif





   

CURRENT CODE
----------------

::

    P[blyth@localhost CSGOptiX]$ opticks-f skipahead
    ./CSGOptiX/CSGOptiX7.cu:    curandState rng = sim->rngstate[idx] ;    // TODO: skipahead using an event_id 

    ./qudarap/QSim.cu:    skipahead(jump, &rng);    
         usage from testing code _QSim_propagate_at_multifilm_mutate

    ./qudarap/QRng.cu:Simple curand generation with skipahead, no encapsulation. 
    ./qudarap/QRng.cu:__global__ void _QRng_generate(T* uu, unsigned ni, unsigned nv, curandStateXORWOW* r, unsigned long long skipahead_  )
    ./qudarap/QRng.cu:    skipahead( skipahead_, &rng ); 
    ./qudarap/QRng.cu:        //if( id == 0 ) printf("//_QRng_generate id %d v %d u %10.4f  skipahead %d \n", id, v, u, skipahead_  ); 
    ./qudarap/QRng.cu:extern void QRng_generate(dim3 numBlocks, dim3 threadsPerBlock, T* uu, unsigned ni, unsigned nv, curandStateXORWOW* r, unsigned long long skipahead_ )
    ./qudarap/QRng.cu:    printf("//QRng_generate ni %d nv %d skipahead %llu \n", ni, nv, skipahead_ ); 
    ./qudarap/QRng.cu:    _QRng_generate<T><<<numBlocks,threadsPerBlock>>>( uu, ni, nv, r, skipahead_ );
    ./qudarap/QRng.hh:Small *skipahead_event_offsets* are for functionality testing, 
    ./qudarap/QRng.hh:    QRng(unsigned skipahead_event_offset=1) ;  
    ./qudarap/QRng.hh:    template <typename T> void generate(   T* u, unsigned ni, unsigned nv, unsigned long long skipahead_ ) ; 

             


    ./qudarap/qrng.h:    unsigned            skipahead_event_offset ; 
    ./qudarap/qrng.h:    qrng(unsigned skipahead_event_offset_)
    ./qudarap/qrng.h:        skipahead_event_offset(skipahead_event_offset_)
    ./qudarap/qrng.h:    unsigned long long skipahead_ = skipahead_event_offset*event_idx ; 
    ./qudarap/qrng.h:    skipahead( skipahead_, &rng ); 

    ./qudarap/tests/QRngTest.cc:    unsigned long long skipahead_ = 0ull ; 
    ./qudarap/tests/QRngTest.cc:    qr.generate<T>(u->values<T>(), num, skipahead_ );
    ./qudarap/tests/QRngTest.cc:void test_generate_skipahead( QRng& qr, unsigned num_event, unsigned num_item, unsigned num_value, unsigned skipahead_event_offset, const char* reldir )
    ./qudarap/tests/QRngTest.cc:    unsigned long long offset = skipahead_event_offset ; 
    ./qudarap/tests/QRngTest.cc:        unsigned long long skipahead_ = offset*event_index ; 
    ./qudarap/tests/QRngTest.cc:        qr.generate<T>( target, num_item, num_value, skipahead_ );
    ./qudarap/tests/QRngTest.cc:    // *skipahead_event_offset* would normally be estimate of maximum number of random 
    ./qudarap/tests/QRngTest.cc:    // unsigned skipahead_event_offset = 1u ; 
    ./qudarap/tests/QRngTest.cc:    // test_generate_skipahead<float>(qr, num_event, num_item, num_value, skipahead_event_offset, "float" ); 
    ./qudarap/tests/QRngTest.py:    def check_skipahead_shifts(self, offset):
    ./qudarap/tests/QRngTest.py:        For example when using skipaheadstep of 1::
    ./qudarap/tests/QRngTest.py:    t.check_skipahead_shifts(1)

    ./qudarap/QRng.cc:QRng::QRng(unsigned skipahead_event_offset)
    ./qudarap/QRng.cc:    qr(new qrng(skipahead_event_offset)),
    ./qudarap/QRng.cc:void QRng::generate( T* uu, unsigned ni, unsigned nv, unsigned long long skipahead_ )
    ./qudarap/QRng.cc:    QRng_generate<T>(numBlocks, threadsPerBlock, d_uu, ni, nv, qr->rng_states, skipahead_ ); 


    ./g4ok/tests/G4OKTest.cc:In future using curand skipahead WITH_SKIPAHEAD will allow the duplication to be avoided 




INTERMEDIATE EXAMPLE CODE : THAT PERHAPS NEEDS TO BE MOVED IF DONT FIND EQUIVALENT IN QUDARAP
------------------------------------------------------------------------------------------------

Q: Does QRngTest do essentially the same as this ? Do I need lower level qrng_test ? 

::

    ./bin/curand.bash:skipahead
    ./cudarap/tests/curand_skipahead.cu:// nvcc curand_skipahead.cu -std=c++11 -ccbin=/usr/bin/clang -o /tmp/curand_skipahead && /tmp/curand_skipahead 
    ./cudarap/tests/curand_skipahead.cu:curand_skipahead.cu
    ./cudarap/tests/curand_skipahead.cu:    skipahead( skip, &rng_states[id]) ;
    ./cudarap/tests/curand_skipahead.cu:        skipahead( skip, &rng_states[id]) ;
    ./cudarap/tests/curand_skipahead.cu:    ss << "/tmp/curand_skipahead_" << mode << ".npy" ;
    ./cudarap/tests/curand_skipahead.py:    a = np.load("/tmp/curand_skipahead_1.npy")
    ./cudarap/tests/curand_skipahead.py:    b = np.load("/tmp/curand_skipahead_2.npy")

    ./examples/Geant4/CerenkovMinimal/src/RunAction.cc:    //const char* embedded_commandline_extra = "--skipaheadstep 1000" ; // see ~/opticks/notes/issues/raja_repeated_photons.rst 


OLD DEAD CODE
---------------

::

    ./optickscore/Opticks.cc:unsigned Opticks::getSkipAheadStep() const  // --skipaheadstep 1000
    ./optickscore/Opticks.hh:       unsigned             getSkipAheadStep() const ;  // --skipaheadstep 1000

    ./optickscore/OpticksCfg.cc:    m_skipaheadstep(0),     
    ./optickscore/OpticksCfg.cc:   char skipaheadstep[128];
    ./optickscore/OpticksCfg.cc:   snprintf(skipaheadstep,128, 
    ./optickscore/OpticksCfg.cc:"Unsigned int skipaheadstep used with ORng"
    ./optickscore/OpticksCfg.cc:"Default %d ", m_skipaheadstep);
    ./optickscore/OpticksCfg.cc:       ("skipaheadstep",  boost::program_options::value<unsigned>(&m_skipaheadstep), skipaheadstep );
    ./optickscore/OpticksCfg.cc:    return m_skipaheadstep ; 
    ./optickscore/OpticksCfg.hh:     unsigned    m_skipaheadstep ; 

    ./optickscore/OpticksEvent.cc:    m_skipahead(0)
    ./optickscore/OpticksEvent.cc:void OpticksEvent::setSkipAhead(unsigned skipahead)  // TODO: move to unsigned long long 
    ./optickscore/OpticksEvent.cc:    m_skipahead = skipahead ; 
    ./optickscore/OpticksEvent.cc:    return m_skipahead ; 
    ./optickscore/OpticksEvent.hh:       void     setSkipAhead(unsigned skipahead);
    ./optickscore/OpticksEvent.hh:       unsigned        m_skipahead ; 

    ./optickscore/OpticksRun.cc:    unsigned skipaheadstep = m_ok->getSkipAheadStep() ; 
    ./optickscore/OpticksRun.cc:    unsigned skipahead =  tagoffset*skipaheadstep ; 
    ./optickscore/OpticksRun.cc:        << " skipaheadstep " << skipaheadstep
    ./optickscore/OpticksRun.cc:        << " skipahead " << skipahead
    ./optickscore/OpticksRun.cc:    evt->setSkipAhead( skipahead ); // TODO: make configurable + move to ULL

    ./optixrap/OPropagator.cc:    unsigned skipahead = evt->getSkipAhead(); 
    ./optixrap/OPropagator.cc:    m_orng->setSkipAhead(skipahead); 
    ./optixrap/OPropagator.cc:    LOG(LEVEL) << "LAUNCH NOW " << m_ocontext->printDesc() << " skipahead " << skipahead ; 
    ./optixrap/OPropagator.hh:        void     setSkipAhead(unsigned skipahead);

    ./optixrap/ORng.cc:    m_rng_skipahead(0)   
    ./optixrap/ORng.cc:void ORng::setSkipAhead( unsigned skipahead )
    ./optixrap/ORng.cc:    m_rng_skipahead = skipahead ; 
    ./optixrap/ORng.cc:    LOG(fatal) << "skipahead " << skipahead ; 
    ./optixrap/ORng.cc:    m_context["rng_skipahead"]->setUint(m_rng_skipahead) ; 
    ./optixrap/ORng.cc:    return m_rng_skipahead ; 
    ./optixrap/ORng.hh:rtDeclareVariable(unsigned int,  rng_skipahead, , );
    ./optixrap/ORng.hh:      void setSkipAhead(unsigned skipahead); 
    ./optixrap/ORng.hh:      unsigned        m_rng_skipahead ;   

    ./optixrap/cu/generate.cu:// rng_states rng_skipahead
    ./optixrap/cu/generate.cu:    //unsigned long long rng_skipahead_ = 10ull ; 
    ./optixrap/cu/generate.cu:    unsigned long long rng_skipahead_ = rng_skipahead ;   // see ORng.hh
    ./optixrap/cu/generate.cu:    skipahead(rng_skipahead_ , &rng) ;  
    ./optixrap/cu/generate.cu:    //rtPrintf("// rng_skipahead %d  %llu \n", rng_skipahead, rng_skipahead_); 

    ./optixrap/tests/cu/reemissionTest.cu://  rng_states rng_skipahead
    ./optixrap/tests/cu/rngTest.cu://  rng_states rng_skipahead
  



QSimTest rng_sequence_with_skipahead
------------------------------------------

With index 0::

    P[blyth@localhost tests]$ QSimTest__SEvt_index=0 ./QSimTest.sh 
    === ephoton.sh : TEST rng_sequence_with_skipahead : unset environment : will use C++ defaults in quad4::ephoton for p0
    2024-09-25 22:09:49.607 INFO  [137823] [QSimTest::EventConfig@609] [ rng_sequence_with_skipahead
    2024-09-25 22:09:49.607 INFO  [137823] [QSimTest::EventConfig@624] ] rng_sequence_with_skipahead
    2024-09-25 22:09:49.608 INFO  [137823] [QSimTest::main@644]  num 1000000 type 2 subfold rng_sequence_with_skipahead ni_tranche_size 100000 print_id -1
     j     (100000, 16, 16) /tmp/QSimTest/rng_sequence_with_skipahead/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset900000.npy
    seq.shape (1000000, 16, 16) 

    In [1]: seq
    Out[1]: 
    array([[[0.74021935, 0.43845114, 0.51701266, ..., 0.54746926,
             0.6531603 , 0.23023781],
            [0.3388562 , 0.76138884, 0.5456815 , ..., 0.85521436,
             0.48867753, 0.18854636],
            [0.5065246 , 0.02055138, 0.9582228 , ..., 0.74793386,
             0.48760796, 0.31805685],
            ...,
            [0.15299392, 0.327105  , 0.8935202 , ..., 0.93996674,
             0.9458555 , 0.19730906],
            [0.85649884, 0.6574796 , 0.06287431, ..., 0.6235617 ,
             0.96832794, 0.5317995 ],
            [0.90195084, 0.42885613, 0.6744496 , ..., 0.59804755,
             0.8195923 , 0.14472319]],




