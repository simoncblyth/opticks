alignment_options_review
===========================

review what those options are doing
--------------------------------------

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero




--okg4
-----------

Picks the OKG4Test executable, default is probably OKTest without bi-simulation 


--align
----------

::

    [blyth@localhost ana]$ opticks-f isAlign
    ./cfg4/CG4.cc:    m_engine(m_ok->isAlign() ? (CRandomListener*)new CRandomEngine(this) : NULL  ),   // --align
    ./cfg4/CRandomEngine.cc:    bool align_mask = m_ok->isAlign() && m_ok->hasMask() ;
    ./optickscore/Opticks.hh:       bool isAlign() const ; // --align
    ./optickscore/Opticks.cc:bool Opticks::isAlign() const  // --align
    ./okg4/OKG4Mgr.cc:    bool align = m_ok->isAlign();


cfg4/CG4.cc
~~~~~~~~~~~~~~

Brings in m_engine CRandomEngine::

    111 CG4::CG4(OpticksHub* hub)
    112     :
    113     m_log(new SLog("CG4::CG4", "", LEVEL)),
    114     m_hub(hub),
    115     m_ok(m_hub->getOpticks()),
    116     m_run(m_ok->getRun()),
    117     m_cfg(m_ok->getCfg()),
    118     m_ctx(m_ok),
    119     m_engine(m_ok->isAlign() ? (CRandomListener*)new CRandomEngine(this) : NULL  ),   // --align
    120     m_physics(new CPhysics(this)),
    121     m_runManager(m_physics->getRunManager()),


cfg4/CRandomEngine
~~~~~~~~~~~~~~~~~~~~

CRandomEngine isa CLHEP::HepRandomEngine which gets annointed
as the Geant4 engine in CRandomEngine::init with 
CLHEP::HepRandom::setTheEngine.

Canonical m_engine instance is resident of CG4 and is instanciated with it, 
when the --align option is used.

With the engine instanciated standard G4UniformRand calls to get random numbers
are routed via the engine which provides values from precooked sequences generated
by curand for each photon record_id on GPU. 

To provide the appropriate sequence of random numbers for the current photon
it is necessary to call CRandomEngine::preTrack  and the m_ctx referenced needs 
to have the photon record id.

Note that nothing is needed on the GPU side for alignment, which just uses
standard curand to get its randoms. 
During development the below optional macros were used for dumping the random consumption.
GPU dumping is only comprehensible when restricting to single photons.

ALIGN_DEBUG 
WITH_ALIGN_DEV_DEBUG 


--dbgskipclearzero 
--------------------

::

    [blyth@localhost cu]$ opticks-f dbgskipclearzero
    ./cfg4/CSteppingAction.cc:            LOG(debug) << " --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft " ; 
    ./optickscore/Opticks.hh:       bool isDbgSkipClearZero() const ; // --dbgskipclearzero
    ./optickscore/Opticks.cc:bool Opticks::isDbgSkipClearZero() const  // --dbgskipclearzero
    ./optickscore/Opticks.cc:   return m_cfg->hasOpt("dbgskipclearzero");
    ./optickscore/OpticksCfg.cc:       ("dbgskipclearzero",  "debug of the maligned six, see notes/issues/RNG_seq_off_by_one.rst") ;


* :doc:`RNG_seq_off_by_one`

::

    /**
    CSteppingAction::UserSteppingAction
    -------------------------------------

    Action depends on the boolean "done" result of CSteppingAction::setStep.
    When done=true this stops tracking, which happens for absorption and truncation.

    When not done the CProcessManager::ClearNumberOfInteractionLengthLeft is normally
    invoked which results in the RNG for AB and SC being cleared.  
    This forces G4VProcess::ResetNumberOfInteractionLengthLeft for every step, 
    as that matches the way Opticks works `om-cls propagate`
    with AB and SC RNG consumption at every "propagate_to_boundary".

    The "--dbgskipclearzero" option inhibits this zeroing in the case of ZeroSteps
    (which I think happen at boundary reflect or transmit) 

    * hmm is OpBoundary skipped because its usually the winner process ? 
      so the standard G4VDiscreteProcess::PostStepDoIt will do the RNG consumption without assistance ?

    See :doc:`stepping_process_review`

    **/

    void CSteppingAction::UserSteppingAction(const G4Step* step)
    {
        bool done = setStep(step);

        m_g4->postStep();


        if(done)
        {   
            G4Track* track = step->GetTrack();    // m_track is const qualified
            track->SetTrackStatus(fStopAndKill);
        }   
        else
        {
            // guess work for alignment
            // should this be done after a jump ?

            bool zeroStep = m_ctx._noZeroSteps > 0 ;   // usually means there was a jump back 
            bool skipClear = zeroStep && m_ok->isDbgSkipClearZero()  ;  // --dbgskipclearzero

            if(skipClear)
            {   
                LOG(debug) << " --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft " ; 
            }   
            else
            {
                CProcessManager::ClearNumberOfInteractionLengthLeft( m_ctx._process_manager, *m_ctx._track, *m_ctx._step );
            }   

            if(m_ok->hasMask())   // --mask 
            {
                LOG(debug) 
                    << "[--mask] CProcessManager::ClearNumberOfInteractionLengthLeft " 
                    << " preStatus " << CStepStatus::Desc(step->GetPreStepPoint()->GetStepStatus())
                    << " postStatus " << CStepStatus::Desc(step->GetPostStepPoint()->GetStepStatus())
                    ;
            }
        }
    }



--dbgnojumpzero 
------------------

::

    [blyth@localhost issues]$ opticks-f dbgnojumpzero
    ./cfg4/CRandomEngine.cc:        bool dbgnojumpzero = m_ok->isDbgNoJumpZero() ; 
    ./cfg4/CRandomEngine.cc:            << " --dbgnojumpzero " << ( dbgnojumpzero ? "YES" : "NO" )
    ./cfg4/CRandomEngine.cc:        if( dbgnojumpzero )
    ./cfg4/CRandomEngine.cc:            LOG(debug) << "rewind inhibited by option: --dbgnojumpzero " ;   
    ./optickscore/Opticks.hh:       bool isDbgNoJumpZero() const ; // --dbgnojumpzero
    ./optickscore/Opticks.cc:bool Opticks::isDbgNoJumpZero() const  // --dbgnojumpzero
    ./optickscore/Opticks.cc:   return m_cfg->hasOpt("dbgnojumpzero");
    ./optickscore/OpticksCfg.cc:       ("dbgnojumpzero",  "debug of the maligned six, see notes/issues/RNG_seq_off_by_one.rst") ;


::

    338 /**
    339 CRandomEngine::postStep
    340 -------------------------
    341 
    342 This is invoked by CG4::postStep
    343     
    344 Normally without zeroSteps this does nothing 
    345 other than resetting the m_current_step_flat_count to zero.
    346 
    347 When there are zeroSteps the RNG sequence is rewound 
    348 by -m_current_step_flat_count as if the current step never 
    349 happened.
    350 
    351 This rewinding for zeroSteps can be inhibited using 
    352 the --dbgnojumpzero option. 
    353 
    354 **/
    355 
    356 void CRandomEngine::postStep()
    357 {
    358 
    359     if(m_ctx._noZeroSteps > 0)
    360     {
    361         int backseq = -m_current_step_flat_count ;
    362         bool dbgnojumpzero = m_ok->isDbgNoJumpZero() ;
    363         
    364         LOG(debug)
    365             << " _noZeroSteps " << m_ctx._noZeroSteps
    366             << " backseq " << backseq
    367             << " --dbgnojumpzero " << ( dbgnojumpzero ? "YES" : "NO" )
    368             ;
    369             
    370         if( dbgnojumpzero )
    371         {
    372             LOG(debug) << "rewind inhibited by option: --dbgnojumpzero " ;
    373         }   
    374         else
    375         {
    376             jump(backseq);
    377         }   
    378     }   
    379     
    380 
    381     if(m_masked)
    382     {
    383         std::string seq = OpticksFlags::FlagSequence(m_okevt_seqhis, true, m_ctx._step_id_valid + 1  );
    384         m_okevt_pt = strdup(seq.c_str()) ;
    385         LOG(debug) 
    386            << " m_ctx._record_id:  " << m_ctx._record_id
    387            << " ( m_okevt_seqhis: " << std::hex << m_okevt_seqhis << std::dec
    388            << " okevt_pt " << m_okevt_pt  << " ) " 
    389            ;
    390     }      
    391     
    392     m_current_step_flat_count = 0 ;   // (*lldb*) postStep 
    393     
    394     if( m_locseq )




--dbgkludgeflatzero
----------------------


::

    [blyth@localhost issues]$ opticks-f dbgkludgeflatzero
    ./cfg4/CRandomEngine.cc:    m_dbgkludgeflatzero(m_ok->isDbgKludgeFlatZero()), 
    ./cfg4/CRandomEngine.cc:    bool kludge = m_dbgkludgeflatzero 
    ./cfg4/CRandomEngine.cc:    bool kludge = m_dbgkludgeflatzero 
    ./cfg4/CRandomEngine.cc:            << " --dbgkludgeflatzero  "
    ./cfg4/CRandomEngine.hh:        bool                          m_dbgkludgeflatzero ; 
    ./optickscore/Opticks.hh:       bool isDbgKludgeFlatZero() const ; // --dbgkludgeflatzero
    ./optickscore/Opticks.cc:bool Opticks::isDbgKludgeFlatZero() const  // --dbgkludgeflatzero
    ./optickscore/Opticks.cc:   return m_cfg->hasOpt("dbgkludgeflatzero");
    ./optickscore/OpticksCfg.cc:       ("dbgkludgeflatzero",  "debug of the maligned six, see notes/issues/RNG_seq_off_by_one.rst") ;


::

    213 /**
    214 CRandomEngine::flat()
    215 ----------------------
    216 
    217 Returns a random double in range 0..1
    218 
    219 A StepToSmall boundary condition immediately following 
    220 FresnelReflection is special cased to avoid calling _flat(). 
    221 Although apparently the returned value is not used (it being from OpBoundary process)
    222 it is still important to avoid call _flat() and misaligning the sequences.
    223 
    224 **/
    225 
    226 double CRandomEngine::flat()
    227 { 
    228     if(!m_internal) m_location = CurrentProcessName();
    229     assert( m_current_record_flat_count < m_curand_nv );
    230     
    231     
    232 #ifdef USE_CUSTOM_BOUNDARY 
    233     bool kludge = m_dbgkludgeflatzero
    234                && m_current_step_flat_count == 0
    235                && m_ctx._boundary_status == Ds::StepTooSmall
    236                && m_ctx._prior_boundary_status == Ds::FresnelReflection
    237                ;
    238 #else          
    239     bool kludge = m_dbgkludgeflatzero
    240                && m_current_step_flat_count == 0
    241                && m_ctx._boundary_status == StepTooSmall
    242                && m_ctx._prior_boundary_status == FresnelReflection
    243                ;
    244 #endif         
    245 
    246     double v = kludge ? _peek(-2) : _flat() ;
    247 
    248     if( kludge )
    249     {
    250         LOG(debug)
    251             << " --dbgkludgeflatzero  "
    252             << " first flat call following boundary status StepTooSmall after FresnelReflection yields  _peek(-2) value "
    253             << " v " << v 
    254             ;
    255             // actually the value does not matter, its just OpBoundary which is not used 
    256     }       
    257     
    258     m_flat = v ;
    259     m_current_record_flat_count++ ;  // (*lldb*) flat 
    260     m_current_step_flat_count++ ; 
    261     
    262     return m_flat ;
    263 }   





