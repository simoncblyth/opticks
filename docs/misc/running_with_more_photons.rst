running_with_more_photons
============================


Not Enough RNG issue::


    ------------------------- 
    Number of Scintillation Photons:  12177
    Number of Cerenkov Photons:  3543830

    -------->Storing hits in the ROOT file: in this event there are 37929 hits in the tracker chambers:
    HC: volTPCActive_lArTPC_HC

    ###[ EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    2020-11-19 09:41:50.832 FATAL [20312] [OpticksEvent::resize@1100] NOT ENOUGH RNG : USE OPTION --rngmax 3/10/100  num_photons 3556007 rng_max 3000000
    G4OpticksTest: /home/wenzel/gputest/opticks/optickscore/OpticksEvent.cc:1106: void OpticksEvent::resize(): Assertion `enoughRng && " need to prepare and persist more RNG states up to maximual per propagation number"' failed.
    Aborted (core dumped)
    -------------------------- 



Initializing cuRAND requires a large GPU stack size so this is done in separate CUDA launches which 
are done during installation by *cudarap-prepare-installation*

::

    epsilon:opticks blyth$ t opticks-prepare-installation
    opticks-prepare-installation () 
    { 
        local msg="=== $FUNCNAME :";
        echo $msg generating RNG seeds into installcache;
        cudarap-;
        cudarap-prepare-installation
    }
    epsilon:opticks blyth$ cudarap-
    epsilon:opticks blyth$ t cudarap-prepare-installation
    cudarap-prepare-installation () 
    { 
        local size;
        cudarap-prepare-sizes | while read size; do
            CUDARAP_RNGMAX_M=$size cudarap-prepare-rng-;
        done
    }
    epsilon:opticks blyth$ t cudarap-prepare-rng-
    cudarap-prepare-rng- () 
    { 
        local msg="=== $FUNCNAME :";
        local path=$(cudarap-rngpath);
        [ -f "$path" ] && echo $msg path $path exists already && return 0;
        CUDARAP_RNG_DIR=$(cudarap-rngdir) CUDARAP_RNG_MAX=$(cudarap-rngmax) $(cudarap-ibin)
    }


Running *cudarap-prepare-installation* again will just list the curandState files::

    [blyth@localhost ~]$ cudarap-
    [blyth@localhost ~]$ cudarap-prepare-installation
    === cudarap-prepare-rng- : path /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin exists already
    === cudarap-prepare-rng- : path /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin exists already
    === cudarap-prepare-rng- : path /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_10000000_0_0.bin exists already
    [blyth@localhost ~]$ 

Bash functions for creation of larger files are::

    597 cudarap-prepare-rng-400M(){ CUDARAP_RNGMAX_M=400 cudarap-prepare-rng- ; }
    598 cudarap-prepare-rng-200M(){ CUDARAP_RNGMAX_M=200 cudarap-prepare-rng- ; }
    599 cudarap-prepare-rng-100M(){ CUDARAP_RNGMAX_M=100 cudarap-prepare-rng- ; }
    600 cudarap-prepare-rng-10M(){  CUDARAP_RNGMAX_M=10  cudarap-prepare-rng- ; }
    601 cudarap-prepare-rng-2M(){   CUDARAP_RNGMAX_M=2   cudarap-prepare-rng- ; }
    602 cudarap-prepare-rng-1M(){   CUDARAP_RNGMAX_M=1   cudarap-prepare-rng- ; }


The curandState file that is used depends on the **--rngmax** option, from okc/OpticksCfg.cc it is apparent 
that the default **rngmax** is 3M::

     113     m_rngmax(3),
     114     m_rngmaxscale(1000000),


Assuming you have the 10M already saved you can increase the maximum number of photons Opticks can handle with, eg::
   
    --rngmax 10 

It is also possible to change the seed and offset from their defaults of zero with::

    --rngseed 1
    --rngoffset 42
      
If 10M photons is insufficient use the below to initialize more curandState slots, eg for 100M::

    CUDARAP_RNGMAX_M=100 cudarap-prepare-rng-


When using embedded Opticks withing G4Opticks
-----------------------------------------------

Typically the executables command line is not parsed by Opticks when using an 
embedded Opticks as when using G4Opticks.  
Opticks is instanciated when the *G4Opticks::setGeometry* method is called, 
thus to change config of Opticks invoke *G4Opticks::setEmbeddedCommandlineExtra* 
prior to calling *G4Opticks::setGeometry* for example::

    const char* extra = "--rngmax 10 --rngseed 1 --rngoffset 42" ; 
    m_g4ok->setEmbeddedCommandlineExtra(extra);  
    


What is the maximum number of photons that can be handled at once ?
-----------------------------------------------------------------------
     
The maximum is limited by GPU VRAM. Each photon takes 112 bytes: 

* 64 bytes (4*4*4 bytes for 16 32-bit floats/ints) of parameters 
* 48 bytes of curandState.

400M photons corresponding to about 45G has been found to be close to the maximum possible 
when using a 48G VRAM GPU (NVIDIA Quadro RTX 8000).


oxrap/ORng : populates rng_states in the OptiX GPU context
--------------------------------------------------------------

::

    032 /**     
     33 ORng
     34 ====    
     35     
     36 Uploads persisted curand rng_states to GPU.
     37 Canonical instance m_orng is ctor resident of OPropagator.
     38         
     39 Work is mainly done by cudarap-/cuRANDWrapper
     40         
     41 TODO: investigate Thrust based alternatives for curand initialization 
     42       potential for eliminating cudawrap- 
     43 
     44 **/     
    ...
    073 void ORng::init()
     74 {
     75     unsigned rng_max = m_ok->getRngMax();
    ...
         
    110     // OptiX owned RNG states buffer (not CUDA owned)
    111     m_rng_states = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER);
    112 
    113     m_rng_states->setElementSize(sizeof(curandState));
    114 
    115     if(num_mask == 0)
    116     {
    117         m_rng_states->setSize(rng_max);
    118 
    119         curandState* host_rng_states = static_cast<curandState*>( m_rng_states->map() );
    120 
    121         m_rng_wrapper->setItems(rng_max); // why ? to identify which cache file to load i suppose
    122 
    123         m_rng_wrapper->LoadIntoHostBuffer(host_rng_states, rng_max );
    124 
    125         m_rng_states->unmap();
    126     }
    127     else
    128     {
    129         m_rng_states->setSize(num_mask);
    130 
    131         curandState* host_rng_states = static_cast<curandState*>( m_rng_states->map() );
    132 
    133         m_rng_wrapper->setItems(rng_max); // still need to load the full cache
    134 
    135         m_rng_wrapper->LoadIntoHostBufferMasked(host_rng_states, m_mask ) ; // but make partial copy 
    136 
    137         m_rng_states->unmap();
    138     }
    139 
    140     m_context["rng_states"]->setBuffer(m_rng_states);
    141 }
    142 
          


oxrap/OPropagator : instanciates ORng
----------------------------------------

::

     65 OPropagator::OPropagator(Opticks* ok, OEvent* oevt, OpticksEntry* entry)
     66     :
     67     m_log(new SLog("OPropagator::OPropagator","", LEVEL)),
     68     m_ok(ok),
     69     m_oevt(oevt),
     70     m_ocontext(m_oevt->getOContext()),
     71     m_context(m_ocontext->getContext()),
     72     m_orng(new ORng(m_ok, m_ocontext)),
     73     m_propagateoverride(m_ok->getPropagateOverride()),
     74     m_nopropagate(false),
     75     m_entry(entry),
     76     m_entry_index(entry->getIndex()),
     77     m_prelaunch(false),



