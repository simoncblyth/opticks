genstep-workflow-review
==========================

g4ok/G4Opticks
    passes 6*4 parameters over to m_genstep_collector (CGenstepCollector)
    
cfg4/CGenstepCollector

    047 CGenstepCollector::CGenstepCollector(const NLookup* lookup)
     48     :
     49     m_lookup(lookup),
     50     m_genstep(NPY<float>::make(0,6,4)),
     51    // m_gs(new OpticksGenstep(m_genstep)),
     52     m_genstep_itemsize(m_genstep->getNumValues(1)),
     53     m_genstep_values(new float[m_genstep_itemsize]),
     54     m_scintillation_count(0),
     55     m_cerenkov_count(0),
     56     m_machinery_count(0)
     57 {
    ...
    134 void CGenstepCollector::collectScintillationStep
    135 (
    136             G4int                /*id*/,
    137             G4int                parentId,
    138             G4int                materialId,
    139             G4int                numPhotons,
    140 
    ...
    165 )
    166 {
    167      m_scintillation_count += 1 ;   // 1-based index
    168      m_gs_photons.push_back(numPhotons);
    169 
    173      uif_t uifa[4] ;
    174      uifa[0].i = SCINTILLATION ;
    175 
    176     // id == 0 ? m_scintillation_count : id  ;   // use the 1-based index when id zero 
    177      uifa[1].i = parentId ;
    178      uifa[2].i = translate(materialId) ;   // raw G4 materialId translated into GBndLib material line for GPU usage 
    179      uifa[3].i = numPhotons ;
    180 
    ...
    199      float* ss = m_genstep_values ;
    200 
    201      ss[0*4+0] = uifa[0].f ;
    ...
    229      ss[5*4+3] = spare2 ;
    230 
    231      m_genstep->add(ss, m_genstep_itemsize);
    232 }



    226 void CGenstepCollector::collectCerenkovStep
    227 (
    228             G4int              /*id*/,
    229             G4int                parentId,
    230             G4int                materialId,
    231             G4int                numPhotons,
    ...
    257 )
    258 {
    259      m_cerenkov_count += 1 ;   // 1-based index
    260      m_gs_photons.push_back(numPhotons);
    261 
    265      uif_t uifa[4] ;
    266      uifa[0].i = CERENKOV ;
    267    // id == 0 ? -m_cerenkov_count : id  ;   // use the negated 1-based index when id zero 
    268      uifa[1].i = parentId ;
    269      uifa[2].i = translate(materialId) ;
    270      uifa[3].i = numPhotons ;
    ...
    295      float* cs = m_genstep_values ;
    296 
    297      cs[0*4+0] = uifa[0].f ;
    ...
    325      cs[5*4+3] = postVelocity ;
    326 
    327      m_genstep->add(cs, m_genstep_itemsize);
    328 }



* note the *id* parameter is not used in either of the above, 
  that slot is currently set to SCINTILLATION or CERENKOV   (enum from optickscore/OpticksPhoton.h)

* float encoded param added to the m_genstep NPY<float>  (n,6,4) 




okop/G4Opticks passes gensteps to okop/OpMgr 
----------------------------------------------

::

    378 int G4Opticks::propagateOpticalPhotons()
    379 {
    380     m_gensteps = m_genstep_collector->getGensteps();
    381     const char* gspath = m_ok->getDirectGenstepPath();
    382 
    383     LOG(info) << " saving gensteps to " << gspath ;
    384     m_gensteps->setArrayContentVersion(G4VERSION_NUMBER);
    385     m_gensteps->save(gspath);
    386 
    387     // initial generated photons before propagation 
    388     // CPU genphotons needed only while validating 
    389     m_genphotons = m_g4photon_collector->getPhoton();
    390     m_genphotons->setArrayContentVersion(G4VERSION_NUMBER);
    391 
    392     //const char* phpath = m_ok->getDirectPhotonsPath(); 
    393     //m_genphotons->save(phpath); 
    394 
    395 
    396     if(m_gpu_propagate)
    397     {
    398         m_opmgr->setGensteps(m_gensteps);
    399         m_opmgr->propagate();     // GPU simulation is done in here 
    400 



okop/OpMgr hands on to m_run
---------------------------------

::

    107 void OpMgr::propagate()
    108 {
    109     LOG(LEVEL) << "\n\n[[\n\n" ;
    110 
    111     const Opticks& ok = *m_ok ;
    112    
    113     if(ok("nopropagate")) return ;
    114 
    115     assert( ok.isEmbedded() );
    116 
    117     assert( m_gensteps );
    118 
    119     bool production = m_ok->isProduction();
    120 
    121     bool compute = true ;
    122 
    123     m_gensteps->setBufferSpec(OpticksEvent::GenstepSpec(compute));
    124 
    125     m_run->createEvent(0);
    126 
    127     m_run->setGensteps(m_gensteps);
    128 
    129     m_propagator->propagate();




optickscore/OpticksRun imports the gensteps
----------------------------------------------


::

    197 void OpticksRun::setGensteps(NPY<float>* gensteps)
    198 {
    199     OK_PROFILE("_OpticksRun::setGensteps");
    200     assert(m_evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");
    201 
    202     if(!gensteps) LOG(fatal) << "NULL gensteps" ;
    203     assert(gensteps);
    204 
    205     LOG(LEVEL) << "gensteps " << gensteps->getShapeString() ;
    206 
    207     m_gensteps = gensteps ;
    208 
    209     importGensteps();
    210     OK_PROFILE("OpticksRun::setGensteps");
    211 }
    212 
    ...
    231 void OpticksRun::importGensteps()
    232 {
    233     OK_PROFILE("_OpticksRun::importGensteps");
    234 
    235     const char* oac_label = m_ok->isEmbedded() ? "GS_EMBEDDED" : NULL ;
    236 
    237     m_g4step = importGenstepData(m_gensteps, oac_label) ;
    238 
    239 
    240     if(m_g4evt)
    241     {
    242         bool progenitor=true ;
    243         m_g4evt->setGenstepData(m_gensteps, progenitor);
    244     }
    245 
    246     m_evt->setGenstepData(m_gensteps);
    247 
    248 
    249 
    ...
    352 G4StepNPY* OpticksRun::importGenstepData(NPY<float>* gs, const char* oac_label)
    353 {
    354     OK_PROFILE("_OpticksRun::importGenstepData");
    355     NMeta* gsp = gs->getParameters();
    356     m_parameters->append(gsp);
    357 
    358     gs->setBufferSpec(OpticksEvent::GenstepSpec(m_ok->isCompute()));
    359 
    360     // assert(m_g4step == NULL && "OpticksRun::importGenstepData can only do this once ");
    361     G4StepNPY* g4step = new G4StepNPY(gs);
    362 
    363     OpticksActionControl oac(gs->getActionControlPtr());
    ...
    378     if(oac("GS_LEGACY"))
    379     {
    380         translateLegacyGensteps(g4step);
    381     }
    382     else if(oac("GS_EMBEDDED"))
    383     {
    384         g4step->addAllowedGencodes( CERENKOV, SCINTILLATION) ;
    385         LOG(LEVEL) << " GS_EMBEDDED collected direct gensteps assumed translated at collection  " << oac.description("oac") ;
    386     }
    387     else if(oac("GS_TORCH"))
    388     {
    389         g4step->addAllowedGencodes(TORCH);
    390         LOG(LEVEL) << " checklabel of torch steps  " << oac.description("oac") ;
    391     }
    392     else if(oac("GS_FABRICATED"))
    393     {
    394         g4step->addAllowedGencodes(FABRICATED);
    395     }
    396     else if(oac("GS_EMITSOURCE"))
    397     {
    398         g4step->addAllowedGencodes(EMITSOURCE);
    399     }
    400     else
    401     {
    402         LOG(LEVEL) << " checklabel of non-legacy (collected direct) gensteps  " << oac.description("oac") ;
    403         g4step->addAllowedGencodes(CERENKOV, SCINTILLATION, EMITSOURCE);
    404     }
    405     g4step->checkGencodes();
    406 
    407     g4step->countPhotons();
    408 
    409     LOG(LEVEL)
    410          << " Keys "
    411          << " TORCH: " << TORCH
    412          << " CERENKOV: " << CERENKOV
    413          << " SCINTILLATION: " << SCINTILLATION
    414          << " G4GUN: " << G4GUN
    415          ;
    416 
    417      LOG(LEVEL)
    418          << " counts "
    419          << g4step->description()
    420          ;
    421 
    422 
    423     OK_PROFILE("OpticksRun::importGenstepData");
    424     return g4step ;
    425 
    426 }


Hmm need to expand the OpticksGenstep.h to include all these types ?

Recall that the old way of stuffing all into OpticksPhoton is not really 
tenable as running out of bits : plus what you need to know at photon
level is not the same as at genstep level.


DONE : started deconflating photon flags and genstep flags 




npy/G4StepNPY checks the gencodes
-------------------------------------

::

    242 void G4StepNPY::addAllowedGencodes(int gencode1, int gencode2, int gencode3, int gencode4 )
    243 {   
    244     if(gencode1 > -1) m_allowed_gencodes.push_back(gencode1);
    245     if(gencode2 > -1) m_allowed_gencodes.push_back(gencode2);
    246     if(gencode3 > -1) m_allowed_gencodes.push_back(gencode3);
    247     if(gencode4 > -1) m_allowed_gencodes.push_back(gencode4);
    248 }
    249 bool G4StepNPY::isAllowedGencode(unsigned gencode) const
    250 {   
    251     return std::find( m_allowed_gencodes.begin(), m_allowed_gencodes.end() , gencode ) != m_allowed_gencodes.end() ;
    252 }
    253 
    254 void G4StepNPY::checkGencodes()
    255 {
    256     // genstep labels must match  
    257 
    258     unsigned numStep = m_npy->getNumItems();
    259     unsigned mismatch = 0 ;
    260 
    261     for(unsigned int i=0 ; i<numStep ; i++ )
    262     {
    263         int label = m_npy->getInt(i,0u,0u);
    264         bool allowed = label > -1 && isAllowedGencode(unsigned(label)) ;
    265 
    266         if( allowed )
    267         {

    // Still invoked from OpticksRun::translateLegacyGensteps
    331 void G4StepNPY::relabel(int cerenkov_label, int scintillation_label)



DONE : Obvious Extension : Genstep versioning
------------------------------------------------

1. new enum OpticksGenstep.h 
2. id -> gentype for identification
3. make it available as an ini for python


::

     13 
     14 enum
     15 {   
     16     OpticksGenstep_Invalid                  = 0,
     17     OpticksGenstep_G4Cerenkov_1042          = 1,
     18     OpticksGenstep_G4Scintillation_1042     = 2, 
     19     OpticksGenstep_DsG4Cerenkov_r3971       = 3,
     20     OpticksGenstep_DsG4Scintillation_r3971  = 4,
     21     OpticksGenstep_NumType                  = 5
     22 };
     23   

::

    epsilon:optickscore blyth$ cat /usr/local/opticks/build/optickscore/OpticksGenstep_Enum.ini 
    OpticksGenstep_Invalid=0
    OpticksGenstep_G4Cerenkov_1042=1
    OpticksGenstep_G4Scintillation_1042=2
    OpticksGenstep_DsG4Cerenkov_r3971=3
    OpticksGenstep_DsG4Scintillation_r3971=4
    OpticksGenstep_NumType=5
    epsilon:optickscore blyth$ 
    epsilon:optickscore blyth$ 


