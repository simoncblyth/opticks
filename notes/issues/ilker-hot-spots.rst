ilker-hot-spots
================

Report from Ilker : presentation on google docs shows low energy looking clumpy as if lower stats that g4 even with same stats
----------------------------------------------------------------------------------------------------------------------------------

* duplicated randoms/photons maybe ? 


::

    Hello Simon, 

    I hope you are doing well. 

    I have been doing some comparisons with Opticks and
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


Response
----------

Hi Ilker, 


> I have been doing some comparisons with Opticks and G4. 
> I have noticed that there are some artifacts and hot spots show up  on hits
> whenever photon yields are low, if I increase the yields , hits more look like
> G4. I wonder if you ever seen this behavior before? 
 
Thank you for you detailed checking and pointing out this clumpyness 
issue and also for your detailed presentation illustrating the problem.
 
I long ago fixed a similar issue in an earlier iteration of Opticks
using curand skipahead. Apparently that feature from old Opticks
did not survive into the full Opticks reimplementation and 
repetition of curand generated randoms from event to event
was happening. 

My commits from today and yesterday should bring that back.

To understand the issue and the fix I need to explain some 
peculiarities with how randoms are generated in Opticks using 
the curand device API.  

* https://docs.nvidia.com/cuda/curand/device-api-overview.html
 
One problem with using curand is that the curand_init initialization 
of the curandState_t needed by all curand random generation on GPU
requires lots of resources. 

The stack size needed to do curand_init is hugely more that the 
stack size needed for ray tracing and simulation. 
Because of this Opticks does that curand_init for the configured maximum number 
of photons permissable in a single launch only at installation time 
(see bash functions opticks-full-prepare opticks-prepare-installation).
The curandState are persisted into ~/.opticks/rngcache/RNG eg::

    P[blyth@localhost RNG]$ l
    total 2750020
    2148440 -rw-rw-r--. 1 blyth blyth 2200000000 Jan 11  2024 QCurandState_50000000_0_0.bin
          4 drwxrwxr-x. 2 blyth blyth       4096 Jan 11  2024 .
          0 lrwxrwxrwx. 1 blyth blyth         55 Nov 30  2023 QCurandState_200000000_0_0.bin -> ../../rngcache_extra/RNG/QCurandState_200000000_0_0.bin
          0 lrwxrwxrwx. 1 blyth blyth         53 Nov 30  2023 QCurandState_2000000_0_0.bin -> ../../rngcache_extra/RNG/QCurandState_2000000_0_0.bin
          0 lrwxrwxrwx. 1 blyth blyth         55 Nov 30  2023 QCurandState_100000000_0_0.bin -> ../../rngcache_extra/RNG/QCurandState_100000000_0_0.bin
     429688 -rw-rw-r--. 1 blyth blyth  440000000 Oct  7  2022 QCurandState_10000000_0_0.bin
     128908 -rw-rw-r--. 1 blyth blyth  132000000 Oct  7  2022 QCurandState_3000000_0_0.bin
      42972 -rw-rw-r--. 1 blyth blyth   44000000 Oct  7  2022 QCurandState_1000000_0_0.bin
          0 drwxrwxr-x. 3 blyth blyth         17 Sep 14  2019 ..
    P[blyth@localhost RNG]$ 


When Opticks runs those initialized curandState are loaded from file 
and uploaded to GPU (see quadarap/QRng.hh qrng.h).  

That means those states can be used without having to do the expensive 
initialization everytime. The result is that Opticks can generate randoms 
with a much smaller stack size meaning that hugely more GPU threads 
can be active at the same time enabling fast ray tracing and hence simulation. 

BUT that means that without some intervention every event starts 
from the exact same curandState and hence the simulation will consume 
the exact same randoms. 

The way to intervene to avoid repeated randoms is to use the 
curand skipahead API to jump ahead in the random sequence for each photon slot in each event. 
That is done in qudarap/qrng.h:: 

     53 inline QRNG_METHOD void qrng::get_rngstate_with_skipahead(curandStateXORWOW& rng, unsigned event_idx, unsigned photon_idx )
     54 {
     55     unsigned long long skipahead_ = skipahead_event_offset*event_idx ;
     56     rng = *(rng_states + photon_idx) ;
     57     skipahead( skipahead_, &rng );
     58 }


The skipahead_event_offset can be configured via envvar OPTICKS_EVENT_SKIPAHEAD
(from sysrap/SEventConfig) the default is current 100,000 and the 
event_idx comes from the Geant4 eventID.  This aims to prevent repetition of 
randoms consumed in the same photon slot in consequtive events. 


> I also noticed that the hit counts agree much better when i turn off the
> reflections on the steel. I introduce 20%  surface reflections on steel by
> using REFLECTIVITY property.
 

What the simulation does depends on both randoms from Geant4 that inflence
the generated gensteps and randoms from Opticks. 
Prior to my fix the Opticks randoms were repeating without manual skipahead. 

To the extent that less randoms are used eg specular reflection doesnt consume
randoms you might expect things to be less messed up.  But its better 
to fix one issue at a time and check after each fix as things are 
too complicated to reason with this way with any reliability. Thats why we 
use simulation, because its too complicated to do things analytically.  

Conversely if it is necessary to check the simulation in 
great detail then you need to use exceedingly simple geometry
and patterns of photons such that you can know exactly 
what should be happen. 


> Also i am curious if there is a way of checking boundary overlaps in Opticks
> and how could i do a good geometry comparison? I have been doing indirectly by
> looking at final position of photons and compare it both G4 and Opticks 
> yet maybe there is an alternative way you may know.
 
Comparing A:Opticks and B:Geant4 simulations when using input photons 
(ie the exact same CPU generated photons in both A and B) is a powerful 
way to find geometry and other issues.  

The so called "record" array records every step point of the photon history. 
This detailed step history can also be recorded from the Geant4 side
using the U4Recorder, allowing recording of the photon histories 
from Geant4 within Opticks SEvt format NumPy arrays. 

Statistical comparisons between the A and B NumPy arrays is the 
first thing to do for validation. 

Going further it is possible to arrange for Geant4 to provide 
the same set of precooked randoms that curand generates 
(by replacing the Geant4 "engine" see u4/U4Random.hh) 
I call that aligned running : it means that scatters, reflections, transmissions
all happen at same places between the simulations. 
So the resulting arrays can be compared directly, unclouded by statistics.  

> Here is the presentation to the plots and some comparisons I have done.  If you
> have any suggestions , i really appreciate.
 
Thank you for the detailed comparisons. 

I am very interested to see those same plots after updating to the 
latest bitbucket Opticks. You might also check the effect 
as you vary the below envvar.

    export OPTICKS_EVENT_SKIPAHEAD=0    
          ## at zero, I expect you should get exactly the same as you presented
          ## already with the clumping from duplicated randoms

    export OPTICKS_EVENT_SKIPAHEAD=100000
          ## at 100k I expect you should match Geant4
          ## what value is actually needed depends on the complexity of the simulation
          ## Its essentially a guess that most photons slots can 
          ## be simulated while consuming less than that number of randoms


Regarding performance, I recently compared ray trace performance between 
the generations:

* 1st gen RTX : NVIDIA TITAN RTX  (Released: Dec 2018)
* 3rd gen RTX : NVIDIA RTX 5000 Ada Generation  (Released: August 2023)

3rd gen is consistently giving a factor of at least 4 faster than 1st gen, 
which appears to confirm the NVIDIA claim of 2x raw ray trace performance
improvement between generations. 

Simon









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




    P[blyth@localhost tests]$ QSimTest__rng_sequence_with_skipahead__eventID=0 ./QSimTest.sh 

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






    P[blyth@localhost tests]$ QSimTest__rng_sequence_with_skipahead__eventID=1 ./QSimTest.sh 

    In [1]: seq
    Out[1]: 
    array([[[0.62425894, 0.72555834, 0.4597076 , ..., 0.42475298,
             0.9087321 , 0.24526083],
            [0.61745024, 0.6575631 , 0.40118492, ..., 0.5730591 ,
             0.8195803 , 0.64033407],
            [0.6623076 , 0.9948266 , 0.7494343 , ..., 0.66807216,
             0.93622154, 0.8137269 ],
            ...,
            [0.21921866, 0.4446953 , 0.7209938 , ..., 0.23494406,
             0.34502968, 0.29158133],
            [0.11280641, 0.69127923, 0.49338955, ..., 0.41528183,
             0.84367394, 0.04531961],
            [0.9738698 , 0.05600067, 0.95384246, ..., 0.21428709,
             0.71177804, 0.38997704]],

           [[0.48497516, 0.68430644, 0.59561753, ..., 0.9785536 ,
             0.67713684, 0.43304485],
            [0.0484974 , 0.550004  , 0.30943045, ..., 0.87187475,
             0.36493173, 0.9270784 ],
            [0.789175  , 0.9536335 , 0.81837696, ..., 0.3339379 ,
             0.9447384 , 0.12426154],
            ...,


Because skipahead zero are back to zeroth::

    P[blyth@localhost tests]$ QSimTest__rng_sequence_with_skipahead__eventID=1 OPTICKS_EVENT_SKIPAHEAD=0 ./QSimTest.sh

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

           [[0.9209938 , 0.46036443, 0.33346406, ..., 0.82454693,
             0.5270629 , 0.9301316 ],
            [0.16302098, 0.7851588 , 0.9419476 , ..., 0.49194995,
             0.5426917 , 0.9343928 ],
            [0.47857913, 0.4494259 , 0.12570204, ..., 0.04226144,
             0.37903434, 0.71457326],
            ...,


Using skipahead of one can see that have shifted the randoms by one consumption:: 

    QSimTest__rng_sequence_with_skipahead__eventID=1 OPTICKS_EVENT_SKIPAHEAD=1 ./QSimTest.sh


    In [1]: seq
    Out[1]: 
    array([[[0.43845114, 0.51701266, 0.15698862, ..., 0.6531603 ,
             0.23023781, 0.3388562 ],
            [0.76138884, 0.5456815 , 0.9702965 , ..., 0.48867753,
             0.18854636, 0.5065246 ],
            [0.02055138, 0.9582228 , 0.7742287 , ..., 0.48760796,
             0.31805685, 0.71192294],
            ...,
            [0.327105  , 0.8935202 , 0.97141856, ..., 0.9458555 ,
             0.19730906, 0.85649884],
            [0.6574796 , 0.06287431, 0.12924866, ..., 0.96832794,
             0.5317995 , 0.90195084],
            [0.42885613, 0.6744496 , 0.8609608 , ..., 0.8195923 ,
             0.14472319, 0.4973046 ]],

           [[0.46036443, 0.33346406, 0.37252042, ..., 0.5270629 ,
             0.9301316 , 0.16302098],
            [0.7851588 , 0.9419476 , 0.4709592 , ..., 0.5426917 ,
             0.9343928 , 0.47857913],
            [0.4494259 , 0.12570204, 0.5727265 , ..., 0.37903434,
             0.71457326, 0.8066413 ],
            ...,








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

           [[0.9209938 , 0.46036443, 0.33346406, ..., 0.82454693,
             0.5270629 , 0.9301316 ],
            [0.16302098, 0.7851588 , 0.9419476 , ..., 0.49194995,
             0.5426917 , 0.9343928 ],
            [0.47857913, 0.4494259 , 0.12570204, ..., 0.04226144,
             0.37903434, 0.71457326],
            ...,





    In [1]: seq
    Out[1]: 
    array([[[0.7483502 , 0.34293526, 0.88547647, ..., 0.5847951 ,
             0.79113036, 0.23981711],
            [0.15539935, 0.7188528 , 0.29173562, ..., 0.5006371 ,
             0.08436476, 0.48330128],
            [0.9787219 , 0.5392139 , 0.6478626 , ..., 0.5202629 ,
             0.51267236, 0.67340326],
            ...,
            [0.27033243, 0.626754  , 0.27585232, ..., 0.07830946,
             0.5084241 , 0.09211873],
            [0.22030236, 0.98426515, 0.6886023 , ..., 0.51349336,
             0.05313287, 0.26358473],
            [0.09876443, 0.50572276, 0.89481217, ..., 0.5351595 ,
             0.57847494, 0.24994943]],

           [[0.90506107, 0.7685051 , 0.0281021 , ..., 0.25146407,
             0.97211236, 0.37875935],
            [0.8918538 , 0.25027007, 0.74765795, ..., 0.11923468,
             0.27575243, 0.47355527],
            [0.38497022, 0.5588296 , 0.39403036, ..., 0.25999963,
             0.57363504, 0.0125184 ],
            ...,




QSimTest needs QEvent for updating of the event index
--------------------------------------------------------


HMM, need to add mock genstep probably::

    2024-09-26 10:41:44.861 INFO  [414965] [QSimTest::main@651]  num 1000000 type 2 subfold rng_sequence_with_skipahead ni_tranche_size 100000 print_id -1
    2024-09-26 10:41:44.861 INFO  [414965] [QSimTest::rng_sequence_with_skipahead@168]  eventID_key QSimTest__rng_sequence_with_skipahead__eventID eventID 100
    TODO: change NPX::Make to NPX::ArrayFromData 
    2024-09-26 10:41:44.862 WARN  [414965] [QEvent::setGenstep@194] No gensteps in SEvt::EGPU early exit QEvent::setGenstep 
    2024-09-26 10:41:44.862 ERROR [414965] [QSim::simulate@360]  QEvent::setGenstep ERROR : have event but no gensteps collected : will skip cx.simulate 
    QU::copy_device_to_host ERROR : device pointer is null 
    2024-09-26 10:41:44.862 FATAL [414965] [QEvent::gatherPhoton@548]  QU::copy_device_to_host photon FAILED  evt->photon N evt->num_photon 0

    Thread 1 "QSimTest" received signal SIGINT, Interrupt.
    0x00007ffff70f04fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007ffff70f04fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff76ce6e0 in QEvent::gatherPhoton (this=0x113218f0, p=0x11322c60) at /home/blyth/opticks/qudarap/QEvent.cc:554
    #2  0x00007ffff76ce869 in QEvent::gatherPhoton (this=0x113218f0) at /home/blyth/opticks/qudarap/QEvent.cc:563
    #3  0x00007ffff76d1251 in QEvent::gatherComponent_ (this=0x113218f0, cmp=4) at /home/blyth/opticks/qudarap/QEvent.cc:859
    #4  0x00007ffff76d0f81 in QEvent::gatherComponent (this=0x113218f0, cmp=4) at /home/blyth/opticks/qudarap/QEvent.cc:838
    #5  0x00007ffff74038e8 in SEvt::gather_components (this=0x11321ae0) at /home/blyth/opticks/sysrap/SEvt.cc:3531
    #6  0x00007ffff74044bc in SEvt::gather (this=0x11321ae0) at /home/blyth/opticks/sysrap/SEvt.cc:3617
    #7  0x00007ffff769518b in QSim::simulate (this=0x113216a0, eventID=100, reset_=false) at /home/blyth/opticks/qudarap/QSim.cc:372
    #8  0x000000000040ae91 in QSimTest::rng_sequence_with_skipahead (this=0x7fffffff4440, ni=1000000, ni_tranche_size_=100000) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:170
    #9  0x000000000040d731 in QSimTest::main (this=0x7fffffff4440) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:662
    #10 0x000000000040ddf9 in main (argc=1, argv=0x7fffffff49f8) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:763
    (gdb) 




Need a special config for testing ?::

    (gdb) f 10
    #10 0x000000000040ddf9 in main (argc=1, argv=0x7fffffff49f8) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:763
    763     qst.main(); 
    (gdb) f 9
    #9  0x000000000040d731 in QSimTest::main (this=0x7fffffff4440) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:662
    662         case RNG_SEQUENCE_WITH_SKIPAHEAD:   rng_sequence_with_skipahead(num, ni_tranche_size) ; break ; 
    (gdb) f 8
    #8  0x000000000040ae91 in QSimTest::rng_sequence_with_skipahead (this=0x7fffffff4440, ni=1000000, ni_tranche_size_=100000) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:170
    170     qs->simulate(eventID, reset); 
    (gdb) f 7
    #7  0x00007ffff769518b in QSim::simulate (this=0x113216a0, eventID=100, reset_=false) at /home/blyth/opticks/qudarap/QSim.cc:372
    372     sev->gather(); 
    (gdb) f 6
    #6  0x00007ffff74044bc in SEvt::gather (this=0x11321ae0) at /home/blyth/opticks/sysrap/SEvt.cc:3617
    3617        gather_components(); 
    (gdb) f 5
    #5  0x00007ffff74038e8 in SEvt::gather_components (this=0x11321ae0) at /home/blyth/opticks/sysrap/SEvt.cc:3531
    3531            NP* a = provider->gatherComponent(cmp); 
    (gdb) p provider
    $1 = (const SCompProvider *) 0x113218f0
    (gdb) f 4
    #4  0x00007ffff76d0f81 in QEvent::gatherComponent (this=0x113218f0, cmp=4) at /home/blyth/opticks/qudarap/QEvent.cc:838
    838     NP* a = proceed ? gatherComponent_(cmp) : nullptr ;
    (gdb) p proceed
    $2 = true
    (gdb) f 3
    #3  0x00007ffff76d1251 in QEvent::gatherComponent_ (this=0x113218f0, cmp=4) at /home/blyth/opticks/qudarap/QEvent.cc:859
    859         case SCOMP_PHOTON:    a = gatherPhoton()   ; break ;   
    (gdb) f 2
    #2  0x00007ffff76ce869 in QEvent::gatherPhoton (this=0x113218f0) at /home/blyth/opticks/qudarap/QEvent.cc:563
    563     gatherPhoton(p); 
    (gdb) p p 
    $3 = (NP *) 0x11322c60
    (gdb) f 1
    #1  0x00007ffff76ce6e0 in QEvent::gatherPhoton (this=0x113218f0, p=0x11322c60) at /home/blyth/opticks/qudarap/QEvent.cc:554
    554     if(rc != 0) std::raise(SIGINT) ; 
    (gdb) f 0
    #0  0x00007ffff70f04fb in raise () from /lib64/libpthread.so.0
    (gdb) 


::

     833 NP* QEvent::gatherComponent(unsigned cmp) const
     834 {
     835     LOG(LEVEL) << "[ cmp " << cmp ;
     836     unsigned gather_mask = SEventConfig::GatherComp();
     837     bool proceed = (gather_mask & cmp) != 0 ;
     838     NP* a = proceed ? gatherComponent_(cmp) : nullptr ;
     839     LOG(LEVEL) << "[ cmp " << cmp << " proceed " << proceed << " a " <<  a ;
     840     return a ;
     841 }



