#include <sstream>
#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksFlags.hh"

// cfg4-
#include "CG4.hh"
#include "OpStatus.hh"

#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "CRec.hh"
#include "CStp.hh"
#include "CStep.hh"
#include "CAction.hh"

#include "CBoundaryProcess.hh"
#include "CRecorder.h"

#include "CDebug.hh"
#include "CWriter.hh"
#include "CRecorder.hh"

#include "PLOG.hh"

const char* CRecorder::PRE  = "PRE" ; 
const char* CRecorder::POST = "POST" ; 

CRec* CRecorder::getCRec() const 
{
    return m_crec ;  
}


CRecorder::CRecorder(CG4* g4, CGeometry* geometry, bool dynamic) 
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

   m_slot(0),

   m_decrement_request(0),
   m_decrement_denied(0),
   m_record_truncate(false),
   m_bounce_truncate(false),
   m_topslot_rewrite(0),

   m_step_action(0)
{   
}

void CRecorder::postinitialize()
{
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);
    if(m_dbg) m_dbg->setMaterialBridge( m_material_bridge );
}

void CRecorder::initEvent(OpticksEvent* evt)  // called by CG4::initEvent
{
    m_writer->initEvent(evt);
}



// invoked by CSteppingAction::setStep
#ifdef USE_CUSTOM_BOUNDARY
bool CRecorder::Record(DsG4OpBoundaryProcessStatus boundary_status)
#else
bool CRecorder::Record(G4OpBoundaryProcessStatus boundary_status)
#endif
{    
    m_step_action = 0 ; 

    if(m_ctx._dbgrec)
    LOG(trace) << "CRecorder::Record"
              << " step_id " << m_ctx._step_id
              << " record_id " << m_ctx._record_id
              << " stage " << CStage::Label(m_ctx._stage)
              ;

    // stage is set by CG4Ctx::setStepOptical from CSteppingAction::setStep
    if(m_ctx._stage == CStage::START)
    { 
        const G4StepPoint* pre = m_ctx._step->GetPreStepPoint() ;
        const G4ThreeVector& pos = pre->GetPosition();
        m_crec->setOrigin(pos);  
        m_crec->clearStp();

        zeroPhoton();       // MUST be invoked prior to setBoundaryStatus, resetting photon history state 

        if(m_dbg) m_dbg->Clear();
    }
    else if(m_ctx._stage == CStage::REJOIN )
    {
        if(m_live)
        { 
            decrementSlot();    // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
        }
        else
        {
            m_crec->clearStp(); // rejoin happens on output side, not in the crec CStp list

            // not-zeroing for REJOINders ?
            // bit confused by this ... because REJOINing track will
            // not usually be the one after the original AB one 
            // so will mixing up the wrong tracks ?
            //
            // does this mean the local photon state is just for live mode ?
        }
    }
    else if(m_ctx._stage == CStage::RECOLL )
    {
        m_decrement_request = 0 ;  
    } 


    unsigned preMaterial = m_material_bridge->getPreMaterial(m_ctx._step) ;
    unsigned postMaterial = m_material_bridge->getPostMaterial(m_ctx._step) ;
    setBoundaryStatus( boundary_status, preMaterial, postMaterial);

    bool done = false ; 

    if( m_live )
    {
         assert( 0 && "moved to CRecorderDead.cc" );  
         //done = LiveRecordStep();
    }
    else
    {
         // should have done also ?
         m_crec->add(m_ctx._step, m_ctx._step_id, boundary_status, m_ctx._stage );

         if(m_ctx._dbgrec)
             LOG(info) << "[--dbgrec]" 
                       << " crec.add NumStps " << m_crec->getNumStps() 
                       ; 
    }
    return done ; 
}


void CRecorder::zeroPhoton()
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

    m_slot = 0 ; 
    m_decrement_request = 0 ; 
    m_decrement_denied = 0 ; 
    m_record_truncate = false ; 
    m_bounce_truncate = false ; 
    m_topslot_rewrite = 0 ; 
}

void CRecorder::decrementSlot()  // NB not just live 
{
    m_decrement_request += 1 ; 

    if(m_slot == 0 )
    {
        m_decrement_denied += 1 ; 
        m_step_action |= CAction::DECREMENT_DENIED ; 

        LOG(warning) << "CRecorder::decrementSlot DENIED "
                     << " slot " << m_slot 
                     << " record_truncate " << m_record_truncate 
                     << " bounce_truncate " << m_bounce_truncate 
                     << " decrement_denied " << m_decrement_denied
                     << " decrement_request " << m_decrement_request
                      ;
        return ;
    }
    m_slot -= 1 ; 
}


#ifdef USE_CUSTOM_BOUNDARY
void CRecorder::setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
#else
void CRecorder::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
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



void CRecorder::posttrackWriteSteps()
{
   // CRecorder::posttrackWriteSteps is invoked from CRecorder::posttrack, 
   // once for the primary photon track and then for 0 or more reemtracks
   // via the record_id the info is written onto the correct place 
   // in the photon record buffer

#ifdef USE_CUSTOM_BOUNDARY
    DsG4OpBoundaryProcessStatus prior_boundary_status = Undefined ;
    DsG4OpBoundaryProcessStatus boundary_status = Undefined ;
    DsG4OpBoundaryProcessStatus next_boundary_status = Undefined ;
#else
    G4OpBoundaryProcessStatus prior_boundary_status = Undefined ;
    G4OpBoundaryProcessStatus boundary_status = Undefined ;
    G4OpBoundaryProcessStatus next_boundary_status = Undefined ;
#endif
    bool     done = false  ;  

    assert(!m_live) ;
    unsigned num = m_crec->getNumStps(); 

    if(m_ctx._dbgrec)
    {
        LOG(info) << "CRecorder::posttrackWriteSteps"
                  << " [--dbgrec] "
                  << " num " << num
                  << " m_slot " << m_slot
                   ;

        std::cout << "CRecorder::posttrackWriteSteps stages:"  ;
        for(unsigned i=0 ; i < num ; i++) std::cout << CStage::Label(m_crec->getStp(i)->getStage()) << " " ; 
        std::cout << std::endl ;  
    }

    for(unsigned i=0 ; i < num ; i++)
    {
        m_step_action = 0 ; 

        CStp* stp  = m_crec->getStp(i);
        CStp* next_stp = m_crec->getStp(i+1) ;   // NULL for i = num - 1 

        CStage::CStage_t stage = stp->getStage();
        const G4Step* step = stp->getStep();
        const G4StepPoint* pre  = step->GetPreStepPoint() ; 
        const G4StepPoint* post = step->GetPostStepPoint() ; 

        prior_boundary_status = i == 0 ? Undefined : boundary_status ; 
        boundary_status = stp->getBoundaryStatus() ; 
        next_boundary_status = next_stp ? next_stp->getBoundaryStatus() : Undefined ; 
      
        unsigned premat = m_material_bridge->getPreMaterial(step) ; 
        unsigned postmat = m_material_bridge->getPostMaterial(step) ; 

        CStage::CStage_t postStage = stage == CStage::REJOIN ? CStage::RECOLL : stage  ; // avoid duping the RE 
        unsigned postFlag = OpPointFlag(post, boundary_status, postStage);

        bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS )) != 0 ;
        bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

        bool postSkip = boundary_status == StepTooSmall && !lastPost  ;  

        bool matSwap = next_boundary_status == StepTooSmall ; // is this swapping on the step before it should ?

        // see notes/issues/geant4_opticks_integration/tconcentric_post_recording_has_seqmat_zeros.rst

        if(lastPost)      m_step_action |= CAction::LAST_POST ; 
        if(surfaceAbsorb) m_step_action |= CAction::SURF_ABS ;  
        if(postSkip)      m_step_action |= CAction::POST_SKIP ; 
        if(matSwap)       m_step_action |= CAction::MAT_SWAP ; 

        switch(stage)
        {
            case CStage::START:  m_step_action |= CAction::STEP_START    ; break ; 
            case CStage::REJOIN: m_step_action |= CAction::STEP_REJOIN   ; break ; 
            case CStage::RECOLL: m_step_action |= CAction::STEP_RECOLL   ; break ;
            case CStage::COLLECT:                               ; break ; 
            case CStage::UNKNOWN:assert(0)                      ; break ; 
        } 

        bool first = m_slot == 0 && stage == CStage::START ;

        unsigned u_premat  = matSwap ? postmat : premat ;
        unsigned u_postmat = ( matSwap || postmat == 0 )  ? premat  : postmat ;

        if(first)            u_premat = premat ;   
        // dont allow any matswap for 1st material : see notes/issues/tboolean-box-okg4-seqmat-mismatch.rst

        if(surfaceAbsorb)    u_postmat = postmat ; 


        if(stage == CStage::REJOIN) 
        {
             decrementSlot();   // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
        }

       // as clearStp for each track, REJOIN will always be i=0

        unsigned preFlag = first ? m_ctx._gen : OpPointFlag(pre,  prior_boundary_status, stage) ;

        if(i == 0)
        {
            m_step_action |= CAction::PRE_SAVE ; 
            done = RecordStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE );  
    
            if(done) m_step_action |= CAction::PRE_DONE ; 

            if(!done)
            {
                 done = RecordStepPoint( post, postFlag, u_postmat, boundary_status,       POST );  
                 m_step_action |= CAction::POST_SAVE ; 
                 if(done) m_step_action |= CAction::POST_DONE ; 
            }
        }
        else
        {
            if(!postSkip && !done)
            {
                m_step_action |= CAction::POST_SAVE ; 
                done = RecordStepPoint( post, postFlag, u_postmat, boundary_status, POST );
                if(done) m_step_action |= CAction::POST_DONE ; 
            }
        }

        // huh: changing the inputs ??? confusing ... but are using next step lookahead, 
        stp->setMat(  u_premat, u_postmat );
        stp->setFlag( preFlag,  postFlag );
        stp->setAction( m_step_action );

        bool hard_truncate = (m_step_action & CAction::HARD_TRUNCATE) != 0 ; 

        if(done && !hard_truncate)
        {
            m_writer->writePhoton(post, m_photon);
        }

        if(done) break ; 
    }   // stp loop
}



void CRecorder::posttrack()
{
    // invoked from CTrackingAction::PostUserTrackingAction

    if(m_ctx._dbgrec) LOG(info) << "CRecorder::posttrack" ; 

    if(!m_live)
    { 
        posttrackWriteSteps();
    }

    if(m_dbg)
    {
        m_dbg->posttrack();
    }
}




#ifdef USE_CUSTOM_BOUNDARY
bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label)
#else
bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label)
#endif
{
    // see notes/issues/geant4_opticks_integration/tconcentric_pflags_mismatch_from_truncation_handling.rst
    //
    // NB this is used by both the live and non-live "canned" modes of recording 
    //
    // Formerly at truncation, rerunning this overwrote "the top slot" 
    // of seqhis,seqmat bitfields (which are persisted in photon buffer)
    // and the record buffer. 
    // As that is different from Opticks behaviour for the record buffer
    // where truncation is truncation, a HARD_TRUNCATION has been adopted.

    bool absorb = ( flag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;
    bool miss = ( flag & MISS ) != 0 ;  

    unsigned slot =  m_slot < m_ctx._steps_per_photon  ? m_slot : m_ctx._steps_per_photon - 1 ; // constrain slot to inclusive range (0,_steps_per_photon-1) 

   
    m_record_truncate = slot == m_ctx._steps_per_photon - 1 ;    // hmm not exactly truncate, just top slot 

    if(m_record_truncate) m_step_action |= CAction::RECORD_TRUNCATE ; 

    if(flag == 0)
    {
       assert(0);
       m_photon._badflag += 1 ; 
       m_step_action |= CAction::ZERO_FLAG ; 

       if(!(boundary_status == SameMaterial || boundary_status == Undefined))
            LOG(warning) << " boundary_status not handled : " << OpBoundaryAbbrevString(boundary_status) ; 
    }


    m_photon.add(slot, flag, material);

    if(m_record_truncate && m_photon.is_rewrite_slot() )  // try to overwrite top slot 
    {
        m_topslot_rewrite += 1 ; 

        if(m_ctx._debug || m_ctx._other)
        {
            LOG(info)
                  << ( m_topslot_rewrite > 1 ? CAction::HARD_TRUNCATE_ : CAction::TOPSLOT_REWRITE_ )
                  << " topslot_rewrite " << m_topslot_rewrite
                  << " prior_flag -> flag " 
                  <<   OpticksFlags::Abbrev(m_photon._flag_prior)
                  << " -> " 
                  <<   OpticksFlags::Abbrev(flag)
                  << " prior_mat -> mat " 
                  <<   ( m_photon._mat_prior == 0 ? "-" : m_material_bridge->getMaterialName(m_photon._mat_prior-1, true)  ) 
                  << " -> "
                  <<   ( m_photon._mat == 0       ? "-" : m_material_bridge->getMaterialName(m_photon._mat-1, true)  ) 
                  ;
        } 

        // allowing a single AB->RE rewrite is closer to Opticks
        if(m_topslot_rewrite == 1 && flag == BULK_REEMIT && m_photon._flag_prior  == BULK_ABSORB)
        {
            m_step_action |= CAction::TOPSLOT_REWRITE ; 
        }
        else
        {
            m_step_action |= CAction::HARD_TRUNCATE ; 
            //assert(0);
            return true ; 
        }
    }


   if(flag == BULK_REEMIT) m_photon.scrub_mskhis(BULK_ABSORB)  ;


    unsigned target_record_id = m_dynamic ? 0 : m_ctx._record_id ; 


    m_writer->writeStepPoint(target_record_id, slot, point, flag, material, label);


    if(m_dbg) m_dbg->Collect(point, boundary_status, m_photon );

    m_slot += 1 ;    // m_slot is incremented regardless of truncation, only local *slot* is constrained to recording range


    m_bounce_truncate = m_slot > m_ctx._bounce_max  ;   

    if(m_bounce_truncate) m_step_action |= CAction::BOUNCE_TRUNCATE ; 

    bool done = m_bounce_truncate || m_record_truncate || absorb || miss ;   


    if(done && m_dynamic)
    {
        m_writer->addDynamicRecords();
    }
   
    /* 
    LOG(info) << " m_slot " << m_slot 
              << " m_ctx._steps_per_photon " << m_ctx._steps_per_photon 
              << " slot " << slot 
              << " done " << done
              << " action " << getStepActionString()
              ;
    */

    return done ;    
}



std::string CRecorder::getStepActionString()
{
    return CAction::Action(m_step_action) ;
}




void CRecorder::Summary(const char* msg)
{
    LOG(info) <<  msg << " " << desc() ;
}

void CRecorder::dump(const char* msg)
{
    LOG(info) << msg ; 
    m_crec->dump("CRec::dump");

    if(m_dbg)
    m_dbg->dump("CRecorder::dump");
}

std::string CRecorder::desc() const 
{
    std::stringstream ss ; 
    ss << std::setw(10) << CStage::Label(m_ctx._stage)
       << " evt " << std::setw(7) << m_ctx._event_id
       << " pho " << std::setw(7) << m_ctx._photon_id 
       << " pri " << std::setw(7) << m_ctx._primary_id
       << " ste " << std::setw(4) << m_ctx._step_id 
       << " rid " << std::setw(4) << m_ctx._record_id 
       << " slt " << std::setw(4) << m_slot
       << " pre " << std::setw(7) << CStep::PreGlobalTime(m_ctx._step)
       << " pst " << std::setw(7) << CStep::PostGlobalTime(m_ctx._step)
       << ( m_dynamic ? " DYNAMIC " : " STATIC " )
       ;

   return ss.str();
}

