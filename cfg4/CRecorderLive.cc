/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// this is **NOT THE DEFAULT**  : AND IS DEAD CODE 



CRecorderLive::CRecorderLive(CG4* g4, CGeometry* geometry, bool dynamic) 
   :
   m_g4(g4),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),
   m_photon(),

   m_crec(new CRec(m_g4)),
   m_dbg(m_ctx._dbgrec || m_ctx._dbgseq ? new CDebug(g4, m_photon, this) : NULL),

   m_evt(NULL),
   m_geometry(geometry),
   m_material_bridge(NULL),
   m_dynamic(dynamic),
   m_live(m_ok->hasOpt("liverecorder")),
   m_writer(new CWriter(g4, m_dynamic)),


   m_boundary_status(Undefined),
   m_prior_boundary_status(Undefined),

   m_premat(0),
   m_prior_premat(0),

   m_postmat(0),
   m_prior_postmat(0),


{
}


void CRecorderLive::postinitialize()
{
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);
}




void CRecorderLive::zeroPhoton()
{


} 


// invoked by CSteppingAction::setStep
#ifdef USE_CUSTOM_BOUNDARY
bool CRecorderLive::Record(DsG4OpBoundaryProcessStatus boundary_status)
#else
bool CRecorderLive::Record(G4OpBoundaryProcessStatus boundary_status)
#endif
{    
    assert(m_live);

    m_step_action = 0 ; 

    if(m_ctx._dbgrec)
    LOG(verbose) << "CRecorderLive::Record"
              << " step_id " << m_ctx._step_id
              << " record_id " << m_ctx._record_id
              << " stage " << CStage::Label(m_ctx._stage)
              ;

    // stage is set by CCtx::setStepOptical from CSteppingAction::setStep
    if(m_ctx._stage == CStage::START)
    { 
        zeroPhoton();       // MUST be invoked prior to setBoundaryStatus, resetting photon history state 
    }
    else if(m_ctx._stage == CStage::REJOIN )
    {
        decrementSlot();    // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
    }
    else if(m_ctx._stage == CStage::RECOLL )
    {
        m_decrement_request = 0 ;  
    } 

    unsigned preMaterial = m_material_bridge->getPreMaterial(m_ctx._step) ;
    unsigned postMaterial = m_material_bridge->getPostMaterial(m_ctx._step) ;
    setBoundaryStatus( boundary_status, preMaterial, postMaterial);

    bool done = RecordStep();

    return done ; 
}



void CRecorderLive::zeroPhoton()
{
    m_boundary_status = Undefined ; 
    m_prior_boundary_status = Undefined ; 

    m_premat = 0 ; 
    m_prior_premat = 0 ; 

    m_postmat = 0 ; 
    m_prior_postmat = 0 ; 

  
    m_photon.clear();
    m_photon._c4.uchar_.x = CStep::PreQuadrant(m_ctx._step) ; // initial quadrant 
    m_photon._c4.uchar_.y = 2u ; 
    m_photon._c4.uchar_.z = 3u ; 
    m_photon._c4.uchar_.w = 4u ; 

    m_decrement_request = 0 ; 
    m_decrement_denied = 0 ; 
    m_record_truncate = false ; 
    m_bounce_truncate = false ; 
    m_topslot_rewrite = 0 ; 

    m_slot = 0 ; 
}



#ifdef USE_CUSTOM_BOUNDARY
void CRecorderLive::setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
#else
void CRecorderLive::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
#endif
{
    // this is invoked before RecordStep 
    m_prior_boundary_status = m_boundary_status ; 
    m_prior_premat = m_premat ; 
    m_prior_postmat = m_postmat ; 

    m_boundary_status = boundary_status ; 
    m_premat = premat ; 
    m_postmat = postmat ;
}






bool CRecorderLive::RecordStep() // this is **NOT THE DEFAULT** 
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


