random_alignment
=====================



CRandomEngine standin to investigate number and position of G4UniformRand flat calls
---------------------------------------------------------------------------------------

::

    tboolean-;tboolean-box --okg4 --align -D

::

    2017-12-04 21:02:54.323 INFO  [208401] [CGenerator::configureEvent@124] CGenerator:configureEvent fabricated TORCH genstep (STATIC RUNNING) 
    2017-12-04 21:02:54.323 INFO  [208401] [CG4Ctx::initEvent@134] CG4Ctx::initEvent photons_per_g4event 10000 steps_per_photon 10 gen 4096
    2017-12-04 21:02:54.323 INFO  [208401] [CWriter::initEvent@69] CWriter::initEvent dynamic STATIC(GPU style) record_max 1 bounce_max  9 steps_per_photon 10 num_g4event 1
    2017-12-04 21:02:54.323 INFO  [208401] [CRec::initEvent@82] CRec::initEvent note recstp
    HepRandomEngine::put called -- no effect!
    2017-12-04 21:02:54.629 INFO  [208401] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 0
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 1
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 2
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 3
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 4
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 5
    2017-12-04 21:02:54.631 INFO  [208401] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2017-12-04 21:02:54.632 INFO  [208401] [CG4::postpropagate@373] CG4::postpropagate(0) ctx CG4Ctx::desc_stats dump_count 0 event_total 1 event_track_count 1
    2017-12-04 21:02:54.632 INFO  [208401] [OpticksEvent::postPropagateGeant4@2040] OpticksEvent::postPropagateGeant4 shape  genstep 1,6,4 nopstep 0,4,4 photon 1,4,4 source 1,4,4 record 1,10,2,4 phosel 1,1,4 recsel 1,10,1,4 sequence 1,1,2 seed 1,1,1 hit 0,4,4 num_photons 1
    2017-12-04 21:02:54.632 INFO  [208401] [OpticksEvent::indexPhotonsCPU@2086] OpticksEvent::indexPhotonsCPU sequence 1,1,2 phosel 1,1,4 phosel.hasData 0 recsel0 1,10,1,4 recsel0.hasData 0
    2017-12-04 21:02:54.632 INFO  [208401] [OpticksEvent::indexPhotonsCPU@2103] indexSequence START 



::

    simon:opticks blyth$ g4-cc HepRandomEngine::put
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:std::ostream & HepRandomEngine::put (std::ostream & os) const {
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:  std::cerr << "HepRandomEngine::put called -- no effect!\n";
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:std::vector<unsigned long> HepRandomEngine::put () const {
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:  std::cerr << "v=HepRandomEngine::put() called -- no data!\n";
    simon:opticks blyth$ vi /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc
    simon:opticks blyth$ 



difficult step : aligning consumption
----------------------------------------

Arrange Opticks code random consumption sequence matches the G4 one
  
This would clearly be impossible(prohibitively expensive to do) 
in a general simulation, but in simulations restricted to 
optical photons (using a subselection of materials/surfaces etc..) 
with initial photons provided as input it seems like it may be possible.

tractable
-----------

get curand to duplicate the GPU sequence on the host, such 
that have random access to per photon slot sequences : these being 
used at startTrack level to feed the sequence into NonRandomEngine

Could just grab from GPU, but would entail wasting a lot of space
as would need to get for every photon the maximum sequence length
that the bounciest truncated photon required.

::

    cudarap/tests/curand_aligned_device.cu
    cudarap/tests/curand_aligned_host.c

* currently the curand host api only working up to slot 4095
* but can just use thrust to random access any slots sequence


TRngBufTest
------------

Produces using curand/thrust 16 floats per photon slot(just example number), 
reproducing the generate.cu in-situ ones from --zrngtest 

Initially attemping to generate 1M at once, got resource issues,
so split the thrust "launches".

::

    simon:tests blyth$ TRngBufTest 
    2017-12-02 20:04:12.284 INFO  [21910] [main@21] TRngBufTest
    TRngBuf::generate ni 100000 id_max 1000
    TRngBuf::generate seq 0 id_offset          0 id_per_gen       1000 remaining     100000
    TRngBuf::generate seq 1 id_offset       1000 id_per_gen       1000 remaining      99000
    TRngBuf::generate seq 2 id_offset       2000 id_per_gen       1000 remaining      98000
    ...
    TRngBuf::generate seq 96 id_offset      96000 id_per_gen       1000 remaining       4000
    TRngBuf::generate seq 97 id_offset      97000 id_per_gen       1000 remaining       3000
    TRngBuf::generate seq 98 id_offset      98000 id_per_gen       1000 remaining       2000
    TRngBuf::generate seq 99 id_offset      99000 id_per_gen       1000 remaining       1000
    (100000, 4, 4)
    [[[ 0.74021935  0.43845114  0.51701266  0.15698862]
      [ 0.07136751  0.46250838  0.22764327  0.32935849]
      [ 0.14406531  0.18779911  0.91538346  0.54012483]
      [ 0.97466087  0.54746926  0.65316027  0.23023781]]

     [[ 0.9209938   0.46036443  0.33346406  0.37252042]
      [ 0.48960248  0.56727093  0.07990581  0.23336816]
      [ 0.50937784  0.08897854  0.00670976  0.95422709]
      [ 0.54671133  0.82454693  0.52706289  0.93013161]]

     [[ 0.03902049  0.25021473  0.18448432  0.96242225]
      [ 0.5205546   0.93996495  0.83057821  0.40973285]
      [ 0.08162197  0.80677092  0.69528568  0.61770737]
      [ 0.25633496  0.21368156  0.34242383  0.22407883]]

     ..., 
     [[ 0.81814659  0.20170131  0.54593664  0.04129851]
      [ 0.38002208  0.91853744  0.02320537  0.05250723]
      [ 0.11425403  0.77515221  0.40338024  0.97540855]
      [ 0.46321765  0.80014837  0.65215546  0.73192346]]

     [[ 0.62748933  0.05319326  0.34443355  0.8561672 ]
      [ 0.2001164   0.3857657   0.31989732  0.40597615]
      [ 0.45497316  0.97913557  0.64739084  0.81499505]
      [ 0.82874513  0.009322    0.81717068  0.57686758]]

     [[ 0.91401154  0.44032493  0.94783556  0.09001808]
      [ 0.9587481   0.98795038  0.2274524   0.04384946]
      [ 0.77744925  0.50308371  0.30509573  0.18650141]
      [ 0.32255048  0.73956126  0.63323611  0.65263885]]]
    simon:tests blyth$ 

::

    In [1]: import os, numpy as np

    In [2]: c = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))

    In [3]: a = np.load("/tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy")

    In [4]: np.all( a == c )
    Out[4]: True



curand aligned with G4 random ?
------------------------------------

Suspect getting different imps of generators
to provide same sequences, would be an exercise in frustration.
And in any case the way curand works, having a "cursor" for each 
photon slot to allow parallel usage means that need to 
operate slot-by-slot.
  
But Geant4 has a NonRandomEngine, which enables
the sequence to be provided as input, see cfg4/tests/G4UniformRandTest.cc 

* reemission would be a complication, because its done all in one go
  with Opticks but in two(or more) separate tracks with Geant4


review G4 random
------------------

::

   g4-;g4-cls Randomize
   g4-;g4-cls Random
   g4-;g4-cls RandomEngine
   g4-;g4-cls NonRandomEngine
   g4-;g4-cls JamesRandom


::

    simon:Random blyth$ grep public\ HepRandomEngine *.*

    DualRand.h:class DualRand: public HepRandomEngine {
    JamesRandom.h:class HepJamesRandom: public HepRandomEngine {
    MTwistEngine.h:class MTwistEngine : public HepRandomEngine {
    MixMaxRng.h:class MixMaxRng: public HepRandomEngine {
    NonRandomEngine.h:class NonRandomEngine : public HepRandomEngine {
    RanecuEngine.h:class RanecuEngine : public HepRandomEngine {
    Ranlux64Engine.h:class Ranlux64Engine : public HepRandomEngine {
    RanluxEngine.h:class RanluxEngine : public HepRandomEngine {
    RanshiEngine.h:class RanshiEngine: public HepRandomEngine {


review curand
----------------


* https://arxiv.org/pdf/1307.5869.pdf
* http://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-overview



feeding sequences to NonRandomEngine
---------------------------------------

::

   g4-;g4-cls NonRandomEngine


cfg4/tests/G4UniformRandTest.cc::

     08 int main(int argc, char** argv)
      9 {   
     10     PLOG_(argc, argv);
     11     
     12     LOG(info) << argv[0] ;
     13 
     14     
     15     unsigned N = 10 ;    // need to provide all that are consumed
     16     std::vector<double> seq ; 
     17     for(unsigned i=0 ; i < N ; i++ ) seq.push_back( double(i)/double(N) );
     18     
     19         
     20     long custom_seed = 9876 ;
     21     //CLHEP::HepJamesRandom* custom_engine = new CLHEP::HepJamesRandom();
     22     //CLHEP::MTwistEngine*   custom_engine = new CLHEP::MTwistEngine();
     23     
     24     CLHEP::NonRandomEngine*   custom_engine = new CLHEP::NonRandomEngine();
     25     custom_engine->setRandomSequence( seq.data(), seq.size() ) ; 
     26     
     27     CLHEP::HepRandom::setTheEngine( custom_engine );
     28     CLHEP::HepRandom::setTheSeed( custom_seed );    // does nothing for NonRandom
     29     
     30     CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;
     31     
     32     
     33     long seed = engine->getSeed() ;
     34     LOG(info) << " seed " << seed 
     35               << " name " << engine->name()
     36             ; 
     37             
     38     for(int i=0 ; i < 10 ; i++)
     39     {
     40         double u = engine->flat() ;   // equivalent to the standardly used: G4UniformRand() 
     41         std::cout << u << std::endl ;
     42     }   
     43     return 0 ;
     44 }   



curand same on host and device
--------------------------------

* https://devtalk.nvidia.com/default/topic/498171/how-to-get-same-output-by-curand-in-cpu-and-gpu/


::

    The quick answer: the simplest way to get the same results on the CPU and GPU
    is to use the host API. This allows you to generate random values into memory
    on the CPU or the GPU, the only difference is whether you call
    curandCreateGeneratorHost() versus curandCreateGenerator().

    To get the same results from the host API and the device API is a bit more
    work, you have to set things up carefully. The basic idea is that
    mathematically there is one long sequence of pseudorandom numbers. This long
    sequence is then cut up into chunks and shuffled together to get a final
    sequence that can be generated in parallel.


trying to get host and device curand to give same results
-----------------------------------------------------------


* matches in slice 0:4096 
* beyond that there is wrap back to the 2nd of 0

* http://docs.nvidia.com/cuda/curand/host-api-overview.html


::

    simon:cudarap blyth$ thrap-print 4095
    thrust_curand_printf
     i0 4095 i1 4096
     id:4095 thread_offset:0 
     0.841588  0.323815  0.475285  0.095566 
     0.397367  0.278207  0.916550  0.810093 
     0.764197  0.476796  0.743895  0.247211 
     0.946511  0.606670  0.736264  0.540743 
    curand_aligned_host
    generate NJ 16 clumps of NI 100000 :  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 
    dump i0:4095 i1:4096 
    i:4095 
    0.841588 0.323815 0.475285 0.095566 
    0.397367 0.278207 0.916550 0.810093 
    0.764197 0.476796 0.743895 0.247211 
    0.946511 0.606670 0.736264 0.540743 
    simon:cudarap blyth$ 
    simon:cudarap blyth$ 
    simon:cudarap blyth$ thrap-print 4096
    thrust_curand_printf
     i0 4096 i1 4097
     id:4096 thread_offset:0 
     0.840685  0.721466  0.500177  0.611869 
     0.970565  0.784008  0.867048  0.428319 
     0.040957  0.309976  0.847280  0.993939 
     0.238374  0.209762  0.010906  0.323518 
    curand_aligned_host
    generate NJ 16 clumps of NI 100000 :  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 
    dump i0:4096 i1:4097 
    i:4096 
    *0.438451* 0.517013 0.156989 0.071368 
    0.462508 0.227643 0.329358 0.144065 
    0.187799 0.915383 0.540125 0.974661 
    0.547469 0.653160 0.230238 0.338856 
    simon:cudarap blyth$ 


    ## beyond 4096 ... getting wrap back 

    simon:cudarap blyth$ thrap-print 0
    thrust_curand_printf
     i0 0 i1 1
     id:   0 thread_offset:0 
     0.740219 *0.438451*  0.517013  0.156989 
     0.071368  0.462508  0.227643  0.329358 
     0.144065  0.187799  0.915383  0.540125 
     0.974661  0.547469  0.653160  0.230238 
    curand_aligned_host
    generate NJ 16 clumps of NI 100000 :  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 
    dump i0:0 i1:1 
    i:0 
    0.740219 0.438451 0.517013 0.156989 
    0.071368 0.462508 0.227643 0.329358 
    0.144065 0.187799 0.915383 0.540125 
    0.974661 0.547469 0.653160 0.230238 
    simon:cudarap blyth$ 







reproduce zrntest subsequences with standanlone thrust_curand_printf
-----------------------------------------------------------------------

Using the known curand_init parameters for each photon_id used by cudarap- machinery 
that prepares the persisted rng_states are able to reproduce zrngtest
subsequences.

thrusttap/tests/thrust_curand_printf.cu::

     05 #include <thrust/for_each.h>
      6 #include <thrust/iterator/counting_iterator.h>
      7 #include <curand_kernel.h> 
      8 #include <iostream> 
      9 #include <iomanip>  
     10 
     11 struct curand_printf
     12 { 
     13     unsigned long long _seed ;
     14     unsigned long long _offset ;
     15     
     16     curand_printf( unsigned long long seed , unsigned long long offset )
     17        :
     18        _seed(seed),
     19        _offset(offset)
     20     {  
     21     }  
     22     
     23     __device__
     24     void operator()(unsigned id)
     25     { 
     26         unsigned int N = 16; // samples per thread 
     27         unsigned thread_offset = 0 ;
     28         curandState s;
     29         curand_init(_seed, id + thread_offset, _offset, &s);
     30         printf(" id:%4u thread_offset:%u \n", id, thread_offset );
     31         for(unsigned i = 0; i < N; ++i) 
     32         { 
     33             float x = curand_uniform(&s);
     34             printf(" %10.4f ", x );  
     35             if( i % 4 == 3 ) printf("\n") ;
     36         }   
     37     }   
     38 };  
     39 
     40 int main(int argc, char** argv)
     41 { 
     42      int id0 = argc > 1 ? atoi(argv[1]) : 0 ;
     43      int id1 = argc > 2 ? atoi(argv[2]) : 1 ;
     44      std::cout
     45          << " id0 " << id0
     46          << " id1 " << id1
     47          << std::endl  
     48          ;  
     49      thrust::for_each(
     50                 thrust::counting_iterator<int>(id0),
     51                 thrust::counting_iterator<int>(id1),
     52                 curand_printf(0,0));
     53     cudaDeviceSynchronize();
     54     return 0;
     55 }   





::

    simon:tests blyth$ thrust_curand_printf 0 1
     id0 0 id1 1
     id:   0 thread_offset:0 
         0.7402      0.4385      0.5170      0.1570 
         0.0714      0.4625      0.2276      0.3294 
         0.1441      0.1878      0.9154      0.5401 
         0.9747      0.5475      0.6532      0.2302 

    simon:tests blyth$ thrust_curand_printf 1 2
     id0 1 id1 2
     id:   1 thread_offset:0 
         0.9210      0.4604      0.3335      0.3725 
         0.4896      0.5673      0.0799      0.2334 
         0.5094      0.0890      0.0067      0.9542 
         0.5467      0.8245      0.5271      0.9301 

    simon:tests blyth$ thrust_curand_printf 2 3
     id0 2 id1 3
     id:   2 thread_offset:0 
         0.0390      0.2502      0.1845      0.9624 
         0.5206      0.9400      0.8306      0.4097 
         0.0816      0.8068      0.6953      0.6177 
         0.2563      0.2137      0.3424      0.2241 


    simon:tests blyth$ thrust_curand_printf 99997 99998 
     id0 99997 id1 99998
     id:99997 thread_offset:0 
         0.8181      0.2017      0.5459      0.0413 
         0.3800      0.9185      0.0232      0.0525 
         0.1143      0.7752      0.4034      0.9754 
         0.4632      0.8001      0.6522      0.7319 

    simon:tests blyth$ thrust_curand_printf 99998 99999 
     id0 99998 id1 99999
     id:99998 thread_offset:0 
         0.6275      0.0532      0.3444      0.8562 
         0.2001      0.3858      0.3199      0.4060 
         0.4550      0.9791      0.6474      0.8150 
         0.8287      0.0093      0.8172      0.5769 

    simon:tests blyth$ thrust_curand_printf 99999 100000
     id0 99999 id1 100000
     id:99999 thread_offset:0 
         0.9140      0.4403      0.9478      0.0900 
         0.9587      0.9880      0.2275      0.0438 
         0.7774      0.5031      0.3051      0.1865 
         0.3226      0.7396      0.6332      0.6526 





    simon:cudarap blyth$ curand_aligned_host 99997 100000
    j:0 generate NI 100000 
    j:1 generate NI 100000 
    j:2 generate NI 100000 
    j:3 generate NI 100000 
    j:4 generate NI 100000 
    j:5 generate NI 100000 
    j:6 generate NI 100000 
    j:7 generate NI 100000 
    j:8 generate NI 100000 
    j:9 generate NI 100000 
    j:10 generate NI 100000 
    j:11 generate NI 100000 
    j:12 generate NI 100000 
    j:13 generate NI 100000 
    j:14 generate NI 100000 
    j:15 generate NI 100000 
    dump i0:99997 i1:100000 
    i:99997 
    0.147038 0.798850 0.013086 0.858024 
    0.647867 0.735839 0.187833 0.655069 
    0.282454 0.655068 0.556091 0.426581 
    0.167576 0.321348 0.079367 0.099285 
    i:99998 
    0.786790 0.184093 0.507811 0.736662 
    0.317718 0.859347 0.905009 0.908526 
    0.860293 0.958224 0.112510 0.483687 
    0.052960 0.573791 0.291022 0.822895 
    i:99999 
    0.483006 0.974604 0.297720 0.621909 
    0.537028 0.619278 0.449021 0.444462 
    0.742229 0.548157 0.034401 0.118713 
    0.313563 0.877223 0.592213 0.742550 
    simon:cudarap blyth$ 




zrngtest : save 16 curand_uniform into photon buffer
--------------------------------------------------

* need to get the below zrngtest subsequences of randoms CPU side, 
  so can feed to NonRandomEngine ?

* hmm perhaps just grab from GPU ? but problem is do not know the 
  maximum number of rands needed for each photon 
  (actually it will depend on the bouncemax truncation configured)


oxrap/cu/generate.cu::

    264 RT_PROGRAM void zrngtest()
    265 {
    266     unsigned long long photon_id = launch_index.x ;
    267     unsigned int photon_offset = photon_id*PNUMQUAD ;
    268 
    269     curandState rng = rng_states[photon_id];
    270 
    271     photon_buffer[photon_offset+0] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    272     photon_buffer[photon_offset+1] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    273     photon_buffer[photon_offset+2] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    274     photon_buffer[photon_offset+3] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    275 
    276     rng_states[photon_id] = rng ;  // suspect this does nothing in my usage
    277 }


This is using saved rng_states cudarap/cuRANDWrapper_kernel.cu::

    093 __global__ void init_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long seed, unsigned long long offset)
    094 {
    ...
    110    int id = blockIdx.x*blockDim.x + threadIdx.x;
    111    if (id >= threads_per_launch) return;
    112 
    113    curand_init(seed, id + thread_offset , offset, &rng_states[id]);
    114 
    115    // not &rng_states[id+thread_offset] as rng_states is offset already in kernel call
    ...
    122 }


seed and offset both zero, from the filenames::

    simon:cfg4 blyth$ l /usr/local/opticks/installcache/RNG/
    total 258696
    -rw-r--r--  1 blyth  staff     450560 Jun 14 16:23 cuRANDWrapper_10240_0_0.bin
    -rw-r--r--  1 blyth  staff  132000000 Jun 14 16:23 cuRANDWrapper_3000000_0_0.bin
    simon:cfg4 blyth$ 


::

    tboolean-;tboolean-box --zrngtest 

    simon:tests blyth$ ls -l /tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy
    -rw-r--r--  1 blyth  wheel  6400080 Dec  2 14:28 /tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy

    tboolean-;TBOOLEAN_TAG=2 tboolean-box --zrngtest 

    simon:cfg4 blyth$ ls -l /tmp/blyth/opticks/evt/tboolean-box/torch/2/ox.npy
    -rw-r--r--  1 blyth  wheel  6400080 Dec  2 14:35 /tmp/blyth/opticks/evt/tboolean-box/torch/2/ox.npy





    simon:cudarap blyth$ curand_aligned_host 0 3
    j:0 generate NI 100000 
    j:1 generate NI 100000 
    j:2 generate NI 100000 
    j:3 generate NI 100000 
    j:4 generate NI 100000 
    j:5 generate NI 100000 
    j:6 generate NI 100000 
    j:7 generate NI 100000 
    j:8 generate NI 100000 
    j:9 generate NI 100000 
    j:10 generate NI 100000 
    j:11 generate NI 100000 
    j:12 generate NI 100000 
    j:13 generate NI 100000 
    j:14 generate NI 100000 
    j:15 generate NI 100000 
    dump i0:0 i1:3 
    i:0 
    0.740219 0.438451 0.517013 0.156989 
    0.071368 0.462508 0.227643 0.329358 
    0.144065 0.187799 0.915383 0.540125 
    0.974661 0.547469 0.653160 0.230238 
    i:1 
    0.920994 0.460364 0.333464 0.372520 
    0.489602 0.567271 0.079906 0.233368 
    0.509378 0.088979 0.006710 0.954227 
    0.546711 0.824547 0.527063 0.930132 
    i:2 
    0.039020 0.250215 0.184484 0.962422 
    0.520555 0.939965 0.830578 0.409733 
    0.081622 0.806771 0.695286 0.617707 
    0.256335 0.213682 0.342424 0.224079 
    simon:cudarap blyth$ 



This shows the reproducibility of the sequences::

    In [1]: import numpy as np

    In [2]: a = np.load("/tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy")

    In [3]: a
    Out[3]: 
    array([[[ 0.74021935,  0.43845114,  0.51701266,  0.15698862],
            [ 0.07136751,  0.46250838,  0.22764327,  0.32935849],
            [ 0.14406531,  0.18779911,  0.91538346,  0.54012483],
            [ 0.97466087,  0.54746926,  0.65316027,  0.23023781]],

           [[ 0.9209938 ,  0.46036443,  0.33346406,  0.37252042],
            [ 0.48960248,  0.56727093,  0.07990581,  0.23336816],
            [ 0.50937784,  0.08897854,  0.00670976,  0.95422709],
            [ 0.54671133,  0.82454693,  0.52706289,  0.93013161]],

           [[ 0.03902049,  0.25021473,  0.18448432,  0.96242225],
            [ 0.5205546 ,  0.93996495,  0.83057821,  0.40973285],
            [ 0.08162197,  0.80677092,  0.69528568,  0.61770737],
            [ 0.25633496,  0.21368156,  0.34242383,  0.22407883]],

           ..., 
           [[ 0.81814659,  0.20170131,  0.54593664,  0.04129851],
            [ 0.38002208,  0.91853744,  0.02320537,  0.05250723],
            [ 0.11425403,  0.77515221,  0.40338024,  0.97540855],
            [ 0.46321765,  0.80014837,  0.65215546,  0.73192346]],

           [[ 0.62748933,  0.05319326,  0.34443355,  0.8561672 ],
            [ 0.2001164 ,  0.3857657 ,  0.31989732,  0.40597615],
            [ 0.45497316,  0.97913557,  0.64739084,  0.81499505],
            [ 0.82874513,  0.009322  ,  0.81717068,  0.57686758]],

           [[ 0.91401154,  0.44032493,  0.94783556,  0.09001808],
            [ 0.9587481 ,  0.98795038,  0.2274524 ,  0.04384946],
            [ 0.77744925,  0.50308371,  0.30509573,  0.18650141],
            [ 0.32255048,  0.73956126,  0.63323611,  0.65263885]]], dtype=float32)

    In [4]: a.min()
    Out[4]: 5.6193676e-07

    In [5]: a.max()
    Out[5]: 0.99999988

    In [6]: a.shape
    Out[6]: (100000, 4, 4)

    In [7]: b = np.load("/tmp/blyth/opticks/evt/tboolean-box/torch/2/ox.npy")

    In [8]: np.all( a == b )
    Out[8]: True





