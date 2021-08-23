optix7-preparation-overview
==============================

Next Steps
-------------

* complete QUDARap/qctx photon generation with gencode generate switch  
* get it to work within cx:CSGOptiXSimulate 
* event handling encapsulated into a QEvent with qevent GPU counterpart 


Accessors / Building
-----------------------

New accessors from opticks/opticks.bash::

    # optix7 expts 
    c(){  cd $(opticks-home)/CSG ; pwd ; }
    cg(){ cd $(opticks-home)/CSG_GGeo ; pwd ; }
    cx(){ cd $(opticks-home)/CSGOptiX ; pwd ; }
    qu(){ qudarap $* ; }

* TODO: make CSG, CSG_GGeo, CSGOptiX into propa Opticks sub-projs 
* TODO: arrange an alternate proj sequence for collective building 


Old accessors from .opticks_config::

    #c(){   cd ~/CSG ; git status ;  }
    #cg(){  cd ~/CSG_GGeo ; git status ;  }
    #cx(){  cd ~/CSGOptiX ; git status ;  }
    #cto(){ open https://github.com/simoncblyth/CSG/commits ; }
    #cxo(){ open https://github.com/simoncblyth/CSGOptiX/commits ; }


Repos now sub folders under Opticks umbrella
-----------------------------------------------

* https://bitbucket.org/simoncblyth/opticks/src/master/CSG/
* https://bitbucket.org/simoncblyth/opticks/src/master/CSG_GGeo/
* https://bitbucket.org/simoncblyth/opticks/src/master/CSGOptiX/

Old Repos
-----------

* https://github.com/simoncblyth/CSG/
* https://github.com/simoncblyth/CSG_GGeo/
* https://github.com/simoncblyth/CSGOptiX/

Migrating old repos under Opticks umbrella
---------------------------------------------

::

    git-export () 
    { 
        local outdir=$1;
        local msg="=== $FUNCNAME :";
        local name=$(basename $PWD);
        local tmpdir=/tmp/$USER/git-export;
        mkdir -p $tmpdir;
        echo $msg name $name into $tmpdir/$name;
        git archive --format=tar --prefix=$name/ HEAD | ( cd_func $tmpdir && tar xf - )
    }

::
    
    cd ~/CSG
    git-export
    rm /tmp/$USER/git-export/CSG/.gitignore
    rm /tmp/$USER/git-export/CSG/README.rst
    cp -r /tmp/$USER/git-export/CSG ~/opticks/

    cd ~/CSG_GGeo
    git-export
    rm /tmp/$USER/git-export/CSG_GGeo/.gitignore
    rm /tmp/$USER/git-export/CSG_GGeo/README.rst
    cp -r /tmp/$USER/git-export/CSG_GGeo ~/opticks/

    cd ~/CSGOptiX
    git-export
    rm /tmp/$USER/git-export/CSGOptiX/.gitignore
    rm /tmp/$USER/git-export/CSGOptiX/README.rst
    cp -r /tmp/$USER/git-export/CSGOptiX ~/opticks/



CMake project dependencies
-----------------------------


::

      CSG : CUDA SysRap 

      CSG_GGeo :  CUDA CSG GGeo  


Previously::

      CSGOptiX : CUDA OpticksCore CSG 

      QudaRap : GGeo OpticksCUDA

         * using GGeo NPY Sysrap
         * aiming to lower to Sysrap only : using NP arrays   

Now::

      QudaRap : SysRap OpticksCUDA

      CSGOptiX : CUDA OpticksCore CSG QudaRap


Where to develop what, General Principal 
------------------------------------------

* do not implement anything in CSGOptiX that can be done in QUDARap or SysRap

Event Handling : follow approach of optixrap/OEvent but no OptiX, QEvent ?
-------------------------------------------------------------------------------

* QEvent encapsulation
* need to make connection with OpticksEvent ? make QUDARap depend on OKC/OpticksEvent 
* OR start with lower level SEvent, built of NP arrays ?


Project summaries
------------------------

CSG
    base model, including CSGFoundary nexus, creates CSGFoundry geometries

CSG_GGeo
    loads GGeo geometry and creates and saves as CSGFoundary 

    * hmm : currently just geometry, no material properties 

    * workflow and Qudarap dependencies can be kept simpler if 
      this translation also collects the things needed 
      from GScintillatorLib and GBndLib for formation of the textures. 
      So can get the GGeo dependency over with in one go.

  
CSGOptiX
    renders CSGFoundary geometry using either OptiX pre-7 or 7 

QUDARap
    pure CUDA photon generation, revolving around GPU side qctx.h 

    * **dependency on GGeo seems a bit out of place**


DONE : Removed QUDARap GGeo,NPY  dependency
-----------------------------------------------------------------

* hmm : GGeo dependency of QUDArap is fairly weak

* bringing the simulation into CSGOptiX means depending on QudaRap, 
  so its beneficial for QudaRap to have few dependencies 

* GGeo is used only in QCtx/QScint/QBnd to access GScintillatorLib and GBndLib for forming 
  scintillation and boundary textures. 

* using lower level types in the interface (think NP rather than GGeo, GScintillatorLib) 
  drastically improved flexibility 

* Direct use of GGeo could be eliminated by changing interface to communicate 
  properties via NP arrays.  

* GGeo is loaded and used in QBndTest QCtxTest 

* hmm adding GGeo dependency for the tests only is a possibility, but its cleaner to 
  load NP from within the persisted CSGFoundry 


Progress
~~~~~~~~~~

* DONE : added "NP* NPY::spawn" to yield NP arrays from NPY ones so can convert NPY 
  coming out of GGeo into NP for holding by CSGFoundry model 

  * use this in CSG_GGeo
  * an alternative is to load NP from persisted NPY : but do not want to assume 
    things have been persisted to file 

* also to remove NPY usage from quadrap/QScint 
  in preparation for CSGFoundry model carrying NP arrays of the properties needed by 
  QScint and QBnd 

* DONE : removed GGeo/GScintillatorLib/GBndLib usage in QudaRap arranging 
  for material props to be passed via NP 

  * but how to do that : do I  want to force use of full CSGFoundry 
    when just want the material/surface props ?  

  * persisted CSGFoundry geometry is really fast and simple, less than 10 .npy arrays 
    in $TMP/CSG_GGeo/CSGFoundry/ so for simplicity of workflow it makes sense to 
    include the bbnd icdf within it and for testing could just load individuals arrays 
    such as $TMP/CSG_GGeo/CSGFoundry/bbnd.npy  
    then qudarap tests would not need to depend on CSGFoundry just its directory path 

  * canonical usage in CSGOptiX has CSGFoundry available, so in that case 
    can directly access the bbnd, icdf  etc.. from CSGFoundry 







TODO
-----

* prototype project structure for integrating QudaRap qctx.h with OptiX 7 running like CSGOptiX 

  * new package name ? CSGQuda? 
  * how to split rendering and simulation functionality : with duplication avoided ?
  * from perusing CSGOptiX.h looks like need to pull off common geometry core : CSGOptiXGeo ? 
  * TODO: effect the split : what is render specific ? 

    * separate pipelines ? PIP::init names the raygen/miss/hitgroup programs, easy split based on the names 
    * separate Param.h ? its simple enough that having common param seems not so problematic
    * tuck rendering stuff into separate struct or just separate methods ?

* DONE : bring CSG, CSG_GGeo and CSGOptiX under opticks umbrella joining QudaRap  
 
  * needs to be standardized, turn tests into ctest etc..


How much of a separation between rendering and simulation ? DECIDED AS LITTLE AS POSSIBLE
---------------------------------------------------------------------------------------------

* raytrace rendering means the ability to save jpg files viewing geometry, it adds no dependencies 
* separation is just for clarity of organization, no strong technical need 


rendering
    viewpoint input yields frame of pixels

simulation
    genstep input yields buffer of photons 



optix 7 rdr/sim separation at what level PIP/SBT or within the raygen function ?  
----------------------------------------------------------------------------------

::

     58 PIP::PIP(const char* ptx_path_)
     59     :
     60     max_trace_depth(2),
     61     num_payload_values(8),
     62     num_attribute_values(4),
     63     pipeline_compile_options(CreatePipelineOptions(num_payload_values,num_attribute_values)),
     64     program_group_options(CreateProgramGroupOptions()),
     65     module(CreateModule(ptx_path_,pipeline_compile_options))
     66 {
     67     init();
     68 }



* at first glance would seem having separate PIP "rdr" "sim" instances seems appropriate as different payload attribute values etc..
  
  * but looks like would add lots of code/complexity 
  * SBT takes pip ctor argument, so separate SBT too ?
  * hmm annoying to need 2nd SBT for teeing up different raygen data : when hardly use that 
  * SBT is primarily for geometry and hence common : is there some way to keep it fully common ? 

* simulation performance is much more critical so will optimize for that anyhow
* the purpose of the rendering is as a visual geometry check of the simulation geometry, 
  which is best served by keeping the sim/rdr branches as close as possible  

* hmm having a single raygen with a param rgmode to switch between rendering and simulation looks 
  very attractive for minimizing code divergence

  * i like the radical simplicity of that approach  
  * my rendering is totally minimal, expect simulation will use more resources  
    so this approach may be fine in long run too 


Prototype thoughts
-----------------------

* new package depending on CSGOptiX and QudaRap ?

  * current thinking is to remove GGeo dependency on QudaRap, instead 
    focus use of GGeo within CSG_GGeo with material properties needed for the 
    reemission and boundary textures and QProp persisted within CSGFoundary 
    model as NP arrays     
  
  * CSGOptiX can then depend on the lowered QudaRap and access all geometry 
    and properties from the CSGFoundry model 

  * advantage is cleaner workflow and dependencies : which means flexible + fast code
    as geometry access/translation happens only once  
 

First objectives for CSGOptiX with QudaRap
-------------------------------------------

* start with purely numerical approach : fabricate a torch genstep and check intersects of 
  generated photons with the optix 7 geometry 

* create planar 2d torch gensteps as an exercise in checking genstep handling 
  and geometry/intersection positions : populate the render frame during simulation 
  with the 3d intersect positions projected onto the input plane of the gensteps
  
  * this should yield 2d renders of planar cuts thru the geometry, in the 
    process checking genstep handling and geometry intersects. Also this 
    benefits from the rendering machinery together with the simulation 
    machinery.  Plus it should much less resource heavy than 3d, making 
    it good for working with complex geometry on laptop GPU.   

* technically how to get access to the qctx "physics context" from optix 7 intersect code ? 
  look at how the geometry data is uploaded 

  * examine the cx optix launch to see how to introduce the qctx ? another param ? 

* CSGOptiX is too render specific need a lower level intermediate struct
  that can be common to both rendering and simulation  

  * current thinking is to not effect much of a split between rendering/simulation, 
    just using raygenmode to make a switch in __raygen__rg   



Creating seed buffer : associating photons to gensteps
----------------------------------------------------------

new way : actually same as previous, just organized in simpler way
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* move general basis functionality into Sysrap::

    SBuf.hh

    (formerly from thrap)
    iexpand.h        
    strided_range.h   

* initial development in thrustrap/tests/iexpand_stridedTest.cu 
  and applying that experience to create focussed functionality in qudarap/QSeed


reviewing the old way
~~~~~~~~~~~~~~~~~~~~~~~

* too dispersed with implementation smeared over cudarap, thrustrap, okop

::

    okop/OpSeeder.cc
    okop/tests/OpSeederTest.cc

    060 void OpSeeder::seedPhotonsFromGensteps()
     61 {
     62     LOG(debug)<<"OpSeeder::seedPhotonsFromGensteps" ;
     63     if( m_ocontext->isCompute() )
     64     {
     65         seedPhotonsFromGenstepsViaOptiX();
     66     }
     67     else if( m_ocontext->isInterop() )
     68     {
     69 #ifdef WITH_SEED_BUFFER
     70         seedComputeSeedsFromInteropGensteps();
     71 #else
     72         seedPhotonsFromGenstepsViaOpenGL();
     73 #endif
     74     }
     75 
     76    // if(m_ok->hasOpt("onlyseed")) exit(EXIT_SUCCESS);
     77 }

    226 /**
    227 OpSeeder::seedPhotonsFromGenstepsImp
    228 --------------------------------------
    229 
    230 1. create TBuf (Thrust buffer accessors) for the two buffers
    231 2. access CPU side gensteps from OpticksEvent
    232 3. check the photon counts from the GPU side gensteps match those from CPU side
    233    (this implies that the event gensteps must have been uploaded to GPU already)
    234 4. create src(photon counts per genstep) and dst(genstep indices) buffer slices
    235    with appropriate strides and offsets 
    236 5. use TBufPair::seedDestination which distributes genstep indices to every photon
    237 
    238 **/
    239 
    240 void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
    241 {
    242     if(m_dbg)
    243     {   
    244         s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs --dbgseed");
    245         s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox --dbgseed");
    246     }
    247 
    248     TBuf tgs("tgs", s_gs, " ");
    249     TBuf tox("tox", s_ox, " ");
    250    
    251 
    252     OpticksEvent* evt = m_ok->getEvent();
    253     assert(evt);
    254 
    255     NPY<float>* gensteps =  evt->getGenstepData() ;
    256 
    257     unsigned num_genstep_values = gensteps->getNumValues(0) ;
    258 
    259     if(m_dbg)
    260     {
    261        LOG(info) << "OpSeeder::seedPhotonsFromGenstepsImp"
    262                  << " gensteps " << gensteps->getShapeString()
    263                  << " num_genstep_values " << num_genstep_values
    264                  ;
    265        tgs.dump<unsigned>("OpSeeder::seedPhotonsFromGenstepsImp tgs.dump --dbgseed", 6*4, 3, num_genstep_values ); // stride, begin, end 
    266     }
    267 
    268 
    269     unsigned num_photons = getNumPhotonsCheck(tgs);
    270 
    271     OpticksBufferControl* ph_ctrl = evt->getPhotonCtrl();
    272 
    ...
    280     // src slice is plucking photon counts from each genstep
    281     // dst slice points at the first value of each item in photon buffer
    282     // buffer size and num_bytes comes directly from CBufSpec
    283     CBufSlice src = tgs.slice(6*4,3,num_genstep_values) ;  // stride, begin, end 
    284 
    285 #ifdef WITH_SEED_BUFFER
    286     tox.zero();   // huh seeding of SEED buffer requires zeroing ahead ?? otherwise get one 0 with the rest 4294967295 ie overrun -1 
    287     CBufSlice dst = tox.slice(1*1,0,num_photons*1*1) ;
    288 #else
    289     CBufSlice dst = tox.slice(4*4,0,num_photons*4*4) ;
    290 #endif
    291 
    292 
    293     bool verbose = m_dbg ;
    294     TBufPair<unsigned> tgp(src, dst, verbose);
    295     tgp.seedDestination();
    296 
    297 #ifdef WITH_SEED_BUFFER
    298     if(m_dbg)
    299     {
    300         tox.dump<unsigned>("OpSeeder::seedPhotonsFromGenstepsImp tox.dump --dbgseed", 1*1, 0, std::min(num_photons,10000u) ); // stride, begin, end 
    301     }
    302 #endif
    303 
    304 }


    037 template <typename T>
     38 void TBufPair<T>::seedDestination()
     39 {  
     40     if(m_verbose)
     41     { 
     42         m_src.Summary("TBufPair<T>::seedDestination (CBufSlice)src");
     43         m_dst.Summary("TBufPair<T>::seedDestination (CBufSlice)dst");
     44     } 
     45       
     46     typedef typename thrust::device_vector<T>::iterator Iterator;
     47   
     48     thrust::device_ptr<T> psrc = thrust::device_pointer_cast((T*)m_src.dev_ptr) ;
     49     thrust::device_ptr<T> pdst = thrust::device_pointer_cast((T*)m_dst.dev_ptr) ;
     50     
     51     strided_range<Iterator> si( psrc + m_src.begin, psrc + m_src.end, m_src.stride );
     52     strided_range<Iterator> di( pdst + m_dst.begin, pdst + m_dst.end, m_dst.stride );
     53 
     54     iexpand( si.begin(), si.end(), di.begin(), di.end() );
     55 
     56 //#define DEBUG 1   
     57 #ifdef DEBUG
     58     std::cout << "TBufPair<T>::seedDestination " << std::endl ;
     59     thrust::copy( di.begin(), di.end(), std::ostream_iterator<T>(std::cout, " ") );
     60     std::cout << "TBufPair<T>::seedDestination " << std::endl ;
     61 #endif
     62 
     63 }
     64 
     65 template class THRAP_API TBufPair<unsigned int> ;
     66 



    022 /**
     23 strided_range.h
     24 ==================
     25 
     26 
     27 Based on /usr/local/env/numerics/thrust/examples/strided_range.cu
     28 
     29 This example illustrates how to make strided access to a range of values
     30 examples::
     31 
     32    strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6] 
     33    strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
     34    strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
     35    ...
     36 
     37 This enables the plucking of photon counts from the GPU side 
     38 genstep buffer, as used by seeding in okop-::
     39 
     40     195 void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
     41     196 {
     42     ...
     43     235     // src slice is plucking photon counts from each genstep
     44     237     // buffer size and num_bytes comes directly from CBufSpec
     45     238     CBufSlice src = tgs.slice(6*4,3,num_genstep_values) ;  // stride, begin, end 
     46     ...
     47 
     48 
     49 **/
     50 
     51 
     52 #include <thrust/iterator/counting_iterator.h>
     53 #include <thrust/iterator/transform_iterator.h>
     54 #include <thrust/iterator/permutation_iterator.h>
     55 #include <thrust/functional.h>
     56 #include <thrust/device_vector.h>
     57 
     58 template <typename Iterator>
     59 class strided_range
     60 {
     61     public:
     62 
     63     typedef typename thrust::iterator_difference<Iterator>::type difference_type;
     64 
     65     struct stride_functor : public thrust::unary_function<difference_type,difference_type>
     66     {
     67         difference_type stride;
     68 
     69         stride_functor(difference_type stride)
     70             : stride(stride) {}
     71 
     72         __host__ __device__
     73         difference_type operator()(const difference_type& i) const
     74         {
     75             return stride * i;




    021 /** 
     22 
     23 iexpand.h
     24 ===========
     25 
     26 Adapted from  /usr/local/env/numerics/thrust/examples/expand.cu 
     27 
     28 Expand an input sequence of counts by replicating indices of each element the number
     29 of times specified by the count values. 
     30 
     31 The element counts are assumed to be non-negative integers.
     32 
     33 Note that the length of the output is equal 
     34 to the sum of the input counts.
     35 
     36 For example::
     37 
     38     iexpand([2,2,2]) -> [0,0,1,1,2,2]  2*0, 2*1, 2*2
     39     iexpand([3,0,1]) -> [0,0,0,2]      3*0, 0*1, 1*2
     40     iexpand([1,3,2]) -> [0,1,1,1,2,2]  1*0, 3*1, 2*2 
     41 
     42 
     43 A more specific example:
     44 
     45 Every optical photon generating genstep (Cerenkov or scintillation) 
     46 specifies the number of photons it will generate.
     47 Applying iexpand to the genstep photon counts produces
     48 an array of genstep indices that is stored into the photon buffer
     49 and provides a reference back to the genstep that produced it.
     50 This reference index is used within the per-photon OptiX 
     51 generate.cu program to access the corresponding genstep 
     52 from the genstep buffer.
     53 
     54 **/
     55 



    080 template <typename InputIterator,
     81           typename OutputIterator>
     82 void iexpand(InputIterator  counts_first,
     83              InputIterator  counts_last,
     84              OutputIterator output_first,
     85              OutputIterator output_last)
     86 {
     87   typedef typename thrust::iterator_difference<InputIterator>::type difference_type;
     88 
     89   difference_type counts_size = thrust::distance(counts_first, counts_last);
     90   difference_type output_size = thrust::distance(output_first, output_last);
     91 
     92 #ifdef DEBUG
     93   std::cout << "iexpand "
     94             << " counts_size " << counts_size
     95             << " output_size " << output_size
     96             << std::endl ;
     97 #endif
     98 
     99 
    100   thrust::device_vector<difference_type> output_offsets(counts_size, 0);
    101 
    102   thrust::exclusive_scan(counts_first, counts_last, output_offsets.begin());
    103 #ifdef DEBUG
    104   print(
    105      " scan the counts to obtain output offsets for each input element \n"
    106      " exclusive_scan of input counts creating output_offsets of transitions \n"
    107      " exclusive_scan is a cumsum that excludes current value \n"
    108      " 1st result element always 0, last input value ignored  \n"
    109      " (output_offsets) \n"
    110    , output_offsets );
    111 
    112   difference_type output_size2 = thrust::reduce(counts_first, counts_last);    // sum of input counts 
    113   assert( output_size == output_size2 );
    114 #endif
    115 
    116   // scatter indices into transition points of output 
    117   thrust::scatter_if
    118     (thrust::counting_iterator<difference_type>(0),
    119      thrust::counting_iterator<difference_type>(counts_size),
    120      output_offsets.begin(),
    121      counts_first,
    122      output_first);
    123 
    124 #ifdef DEBUG
    125   printf(
    126      " scatter the nonzero counts into their corresponding output positions \n"
    127      " scatter_if( first, last, map, stencil, output ) \n"
    128      "    conditionally copies elements from a source range (indices 0:N-1) into an output array according to a map \n"
    129      "    condition dictated by a stencil (input counts) which must be non-zero to be true \n"
    130      "    map provides indices of where to put the indice values in the output  \n"
    131    );
    132 #endif
    133 


* https://stackoverflow.com/questions/16900837/replicate-a-vector-multiple-times-using-cuda-thrust




