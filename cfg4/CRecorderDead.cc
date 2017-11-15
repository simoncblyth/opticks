

bool CRecorder::LiveRecordStep() // this is **NOT THE DEFAULT** 
{
    assert(m_live);

    switch(m_ctx._stage)
    {
        case  CStage::START:  m_step_action |= CAction::STEP_START    ; break ; 
        case  CStage::REJOIN: m_step_action |= CAction::STEP_REJOIN   ; break ; 
        case  CStage::RECOLL: m_step_action |= CAction::STEP_RECOLL   ; break ;
        case  CStage::COLLECT:                               ; break ; 
        case  CStage::UNKNOWN: assert(0)                     ; break ; 
    } 

    const G4StepPoint* pre  = m_ctx._step->GetPreStepPoint() ; 
    const G4StepPoint* post = m_ctx._step->GetPostStepPoint() ; 

    // shunt flags by 1 relative to steps, in order to set the generation code on first step
    // this doesnt miss flags, as record both pre and post at last step    

    unsigned preFlag = m_slot == 0 && m_ctx._stage == CStage::START ? 
                                      m_ctx._gen 
                                   : 
                                      OpPointFlag(pre,  m_prior_boundary_status, m_ctx._stage )
                                   ;

    unsigned postFlag =               OpPointFlag(post, m_boundary_status      , m_ctx._stage );



    bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS)) != 0 ;

    bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    bool preSkip = m_prior_boundary_status == StepTooSmall && m_ctx._stage != CStage::REJOIN  ;  

    bool matSwap = m_boundary_status == StepTooSmall ; 

    unsigned preMat  = matSwap ? m_postmat : m_premat ;

    unsigned postMat = ( matSwap || m_postmat == 0 )  ? m_premat  : m_postmat ;

    if(surfaceAbsorb) postMat = m_postmat ; 

    bool done = false ; 

    // usually skip the pre, but the post becomes the pre at next step where will be taken 
    // 1-based material indices, so zero can represent None
    //
    //   RecordStepPoint records into m_slot (if < m_steps_per_photon) and increments m_slot
    // 

    if(lastPost)      m_step_action |= CAction::LAST_POST ; 
    if(surfaceAbsorb) m_step_action |= CAction::SURF_ABS ;  
    if(preSkip)       m_step_action |= CAction::PRE_SKIP ; 
    if(matSwap)       m_step_action |= CAction::MAT_SWAP ; 


    if(!preSkip)
    {
        m_step_action |= CAction::PRE_SAVE ; 
        done = RecordStepPoint( pre, preFlag, preMat, m_prior_boundary_status, PRE );    // truncate OR absorb
        if(done) m_step_action |= CAction::PRE_DONE ; 
    }

    if(lastPost && !done )
    {
        m_step_action |= CAction::POST_SAVE ; 
        done = RecordStepPoint( post, postFlag, postMat, m_boundary_status, POST ); 
        if(done) m_step_action |= CAction::POST_DONE ; 
    }

    if(done) 
    {
        m_writer->writePhoton(post, m_photon);
       // _seqhis/_seqmat added to m_photon by the RecordStepPoint is written here
       // REJOIN overwrites into record_id recs
    }

    m_crec->add(m_ctx._step, m_ctx._step_id, m_boundary_status, m_premat, m_postmat, preFlag, postFlag, m_ctx._stage, m_step_action );

    return done ;
}


