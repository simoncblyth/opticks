where_mask_running
====================

hmm : how to fast forward to debug single photon ?
------------------------------------------------------

* on CPU jumping to a photon is essential for easy debugging 

* on GPU almost no point, as it dont help much with debugging, 
  can just dump with pindex : but need to do it to match CPU 


Test masked running
---------------------

::

    tboolean-;tboolean-box --mask 0,1,2,3 -D

    tboolean-;tboolean-box --okg4 --align --mask 0,1,2,3 --pindex 0 -D
        ## runs, but needs some effort to make them the same photons as the unmasked sim 

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -D


Where to implement masking ?
----------------------------

* where to apply the mask ? 
* who needs to know about the mask ?

Cannot just mask input photons, as the 
simulation also depends on curandStates etc.. 
and when aligned the precooked rng relevant to a photon slot.

So mask running needs to be aware of unmasked running, just restrict 
operations.

This argues for incorporating an optional mask buffer 
within the OpticksEvent ?

Alternatively the msk could be regarded as a global input living 
withing OpticksDbg ?

* have tried partially hiding the masking inside,  NEmitPhotonsNPY 


Trace OpticksEvent creation 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


ok-/OKMgr::

    073 void OKMgr::propagate()
     74 {
     75     const Opticks& ok = *m_ok ;
     76 
     77     if(ok("nopropagate")) return ;
     78 
    ...
     94     else if(m_num_event > 0)
     95     {
     96         for(int i=0 ; i < m_num_event ; i++)
     97         {
     98             m_run->createEvent(i);
     99 
    100             m_run->setGensteps(m_gen->getInputGensteps());
    101 
    102             m_propagator->propagate();

    123 /**
    124 OpticksRun::setGensteps
    125 ------------------------
    126 
    127 gensteps and maybe source photon data (via aux association) are lodged into m_g4evt
    128 before passing baton (sharing pointers) with m_evt
    129 
    130 **/
    131 void OpticksRun::setGensteps(NPY<float>* gensteps) // THIS IS CALLED FROM VERY HIGH LEVEL IN OKMgr to OKG4Mgr 
    132 {   


okg-/OpticksGen::

    026 OpticksGen::OpticksGen(OpticksHub* hub)
     27    :
     28    m_hub(hub),
     29    m_ok(hub->getOpticks()),
     30    m_cfg(m_ok->getCfg()),
     31    m_ggb(hub->getGGeoBase()),
     32    m_blib(m_ggb->getBndLib()),
     33    m_lookup(hub->getLookup()),
     34    m_torchstep(NULL),
     35    m_fabstep(NULL),
     36    m_input_gensteps(NULL),
     37    m_csg_emit(hub->findEmitter()),
     38    m_emitter(m_csg_emit ? new NEmitPhotonsNPY(m_csg_emit, EMITSOURCE, m_ok->getSeed(), false) : NULL ),
     39    m_input_photons(NULL),
     40    m_source_code( m_emitter ? EMITSOURCE : m_ok->getSourceCode() )
     41 {
       
     51 void OpticksGen::init()
     52 {
     53     if(m_emitter)
     54     {
     55         initFromEmitter();
     56     }
     57     else
     58     {
     59         initFromGensteps();
     60     }
     61 }

     65 void OpticksGen::initFromEmitter()
     66 {
     67     // emitter bits and pieces get dressed up 
     68     // perhaps make a class to do this ?   
     69 
     70     NPY<float>* iox = m_emitter->getPhotons();
     71     setInputPhotons(iox);
     72 
     73     m_fabstep = m_emitter->getFabStep();
     74 
     75     NPY<float>* gs = m_emitter->getFabStepData();
     76     assert( gs );
     77 
     78     gs->setAux((void*)iox); // under-radar association of input photons with the fabricated genstep
     79 
     80     // this gets picked up by OpticksRun::setGensteps 
     81 
     82 
     83     const char* oac_ = "GS_EMITSOURCE" ;
     84 
     85     gs->addActionControl(OpticksActionControl::Parse(oac_));
     86 
     87     OpticksActionControl oac(gs->getActionControlPtr());
     88     setInputGensteps(gs);
     89 
     90     LOG(info) << "OpticksGen::initFromEmitter getting input photons and shim genstep "
     91               << " input_photons " << m_input_photons->getNumItems()
     92               << " oac : " << oac.description("oac")
     93               ;
     94 }





GPU Side 
----------------------

Although not particularly useful for debugging, have to apply 
masking to GPU sim too for the output events to match those from CPU.

All input buffers used by oxrap-/cu/generate.cu 
will need to be masked, directly for photon_id buffers and indirectly for gensteps.

The buffers are created/uploaded with oxrap-/OEvent from the basis OpticksEvent, the
heavy lifting done by OContext. 



genstep_buffer[genstep_offset]
    input gensteps

source_buffer[photon_offset]
    input photons

seed_buffer[photon_id]
    points to a genstep id, for emitconfig running 
    this is probably all zeros : currently genstep still needed for 
    emitconfig just for the gencode of EMITCONFIG

rng_states[photon_id]
    buffer of pre-initialized curandState


::

    087 // input buffers 
     88 
     89 rtBuffer<float4>               genstep_buffer;
     90 rtBuffer<float4>               source_buffer;
     91 #ifdef WITH_SEED_BUFFER
     92 rtBuffer<unsigned>             seed_buffer ; 
     93 #endif
     94 rtBuffer<curandState, 1>       rng_states ;
     95 


     96 // output buffers 
     97 
     98 rtBuffer<float4>               photon_buffer;
     99 #ifdef WITH_RECORD
    100 rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
    101 rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
    102 #endif
    103 



GPU Side implemented 
----------------------------------------------------------------------------------

* provide an OpticksEvent with a fewer input "source" photons, just the masked
* mask the curandStates 


::

    [blyth@localhost opticks]$ opticks-fl Mask | grep optixrap

    ./optixrap/ORng.cc
          fabricates curandStates rng_states OptiX buffer with just the states of the mask photon indices 
    ./optixrap/OContext.cc 
          sets the kernel output logpath with the absolute (ie original) photon index 
    ./optixrap/OEvent.cc
          has m_mask buffer, but not used



CPU Side
----------

configure the mask : just a list of photon indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpticksCfg.cc::

    .833    m_desc.add_options()
     834        ("mask",     boost::program_options::value<std::string>(&m_mask),
     835                     "comma delimited list of photon indices specifying mask selection to apply to both simulations"
     836                     "see OpticksDbg, CInputPhotonSource, CRandomEngine"
     837                     "notes/issues/where_mask_running.rst "
     838                  );


    102 void OpticksDbg::postconfigure()
    103 {
    104    LOG(verbose) << "setting up"  ;
    105    m_cfg = m_ok->getCfg();
    ...
    111    const std::string& mask = m_cfg->getMask() ;
    121    postconfigure( mask, m_mask );
    ...
    134    if(m_mask.size() > 0)
    135    {
    136        m_mask_buffer = NPY<unsigned>::make_from_vec(m_mask);
    137    }
    140 }


Opticks.hh::

    383    public:
    384        // from OpticksDbg --dindex and --oindex options  
    385        // NB these are for cfg4 debugging  (Opticks uses different approach with --pindex option)
    386        NPY<unsigned>* getMaskBuffer() const ;
    387        const std::vector<unsigned>&  getMask() const ;
    388        unsigned getMaskIndex(unsigned idx) const ;  // original pre-masked index OR idx if no mask 
    389        bool hasMask() const ;
    393        bool isMaskPhoton(unsigned record_id) const ;


Mask buffer is passed to NEmitPhotonsNPY::

    035 OpticksGen::OpticksGen(OpticksHub* hub)
     36     :
     37     m_hub(hub),
     38     m_gun(new OpticksGun(hub)),
     39     m_ok(hub->getOpticks()),
     40     m_cfg(m_ok->getCfg()),
     41     m_ggb(hub->getGGeoBase()),
     42     m_blib(m_ggb->getBndLib()),
     43     m_lookup(hub->getLookup()),
     44     m_torchstep(NULL),
     45     m_fabstep(NULL),
     46     m_csg_emit(hub->findEmitter()),
     47     m_dbgemit(m_ok->isDbgEmit()),
     48     m_emitter(m_csg_emit ? new NEmitPhotonsNPY(m_csg_emit, EMITSOURCE, m_ok->getSeed(), m_dbgemit, m_ok->getMaskBuffer(), m_ok->getGenerateOverride() ) : NULL ),
     49     m_input_photons(NULL),

    117 void OpticksGen::initFromEmitterGensteps()
    118 {
    119     // emitter bits and pieces get dressed up 
    120     // perhaps make a class to do this ?   
    121 
    122     NPY<float>* iox = m_emitter->getPhotons();  // these photons maybe masked 
    123     setInputPhotons(iox);
    124 
    125 
    126     m_fabstep = m_emitter->getFabStep();
    127 
    128     NPY<float>* gs = m_emitter->getFabStepData();
    129     assert( gs );
    130 
    131     gs->setAux((void*)iox); // under-radar association of input photons with the fabricated genstep
    132 
    133     // this gets picked up by OpticksRun::setGensteps 
    134 
    135 
    136     const char* oac_ = "GS_EMITSOURCE" ;
    137 
    138     gs->addActionControl(OpticksActionControl::Parse(oac_));
    139 
    140     OpticksActionControl oac(gs->getActionControlPtr());
    141     setLegacyGensteps(gs);
    142 
    143     LOG(LEVEL)
    144         << "getting input photons and shim genstep "
    145         << " --dbgemit " << m_dbgemit
    146         << " input_photons " << m_input_photons->getNumItems()
    147         << " oac : " << oac.description("oac")
    148         ;
    149 }


::

    ./npy/NEmitPhotonsNPY.hpp
    ./npy/NPY.cpp
         Masked running is handled by generating all photons as normal and then making 
         a masked copy of them 




CRandomEngine::preTrack sets up the original curand sequence of randoms using mask index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    603 void CRandomEngine::preTrack()
    604 {   
    605     m_jump = 0 ; 
    606     m_jump_count = 0 ;
    607 
    608     // assert( m_ok->isAlign() );    // not true for tests/CRandomEngineTest 
    609     bool align_mask = m_ok->hasMask() ;
    610     
    611     unsigned use_index ;
    612     if(align_mask)
    613     {   
    614         unsigned mask_index = m_ok->getMaskIndex( m_ctx._record_id );   // "original" index 
    615         use_index = mask_index ; 
    616         run_ucf_script( mask_index ) ;
    617     }
    618     else
    619     {   
    620         use_index = m_ctx._record_id ;
    621     }
    622     
    623     setupCurandSequence(use_index) ;
    624  
    625  
    626     LOG(debug)
    627         << "record_id: "    // (*lldb*) preTrack
    628         << " ctx.record_id " << m_ctx._record_id
    629         << " use_index " << use_index 
    630         << " align_mask " << ( align_mask ? "YES" : "NO" )
    631         ;
    632 }








where mask running
~~~~~~~~~~~~~~~~~~~~

Running on a subselection, picked via a where-mask of in
[blyth@localhost opticks]$ 
ces.
Apply mask to emitconfig photons, and to the rng inputs.

::

    161 CSource* CGenerator::initInputPhotonSource()
    162 {
    163     LOG(info) << "CGenerator::initInputPhotonSource " ;
    164     NPY<float>* inputPhotons = m_hub->getInputPhotons();
    165     NPY<float>* inputGensteps = m_hub->getInputGensteps();
    166     GenstepNPY* gsnpy = m_hub->getGenstepNPY();
    167 
    168     assert( inputPhotons );
    169     assert( inputGensteps );
    170     assert( gsnpy );
    171 
    172     setGensteps(inputGensteps);
    173     setDynamic(false);
    174 
    175     int verbosity = m_ok->isDbgSource() ? 10 : 0 ;
    176     CInputPhotonSource* cips = new CInputPhotonSource( m_ok, inputPhotons, gsnpy, verbosity) ;
    177 
    178     setNumG4Event( cips->getNumG4Event() );
    179     setNumPhotonsPerG4Event( cips->getNumPhotonsPerG4Event() );
    180 
    181     CSource* source  = static_cast<CSource*>(cips);
    182     return source ;
    183 }

::

    013 void CPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
     14 {
     15     m_source->GeneratePrimaryVertex(event);
     16 }

   

    



 
