#include <sstream>
#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

// cfg4-
#include "CG4.hh"
#include "OpStatus.hh"

#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "CRec.hh"

#include "CPoi.hh"
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

std::string CRecorder::getStepActionString()
{
    return CAction::Action(m_state._step_action) ;
}

CRec* CRecorder::getCRec() const 
{
    return m_crec ;  
}

CRecorder::CRecorder(CG4* g4, CGeometry* geometry, bool dynamic) 
   :
   m_g4(g4),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),
   m_recpoi(m_ok->isRecPoi()),
   m_state(m_ctx),
   m_photon(m_ctx, m_state),

   m_crec(new CRec(m_g4)),
   m_dbg(m_ctx.is_dbg() ? new CDebug(g4, m_photon, this) : NULL),

   m_evt(NULL),
   m_geometry(geometry),
   m_material_bridge(NULL),
   m_dynamic(dynamic),
   m_live(false),
   m_writer(new CWriter(g4, m_photon, m_dynamic)),
   m_not_done_count(0)
{   
}

void CRecorder::postinitialize()
{
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);

    m_crec->setMaterialBridge( m_material_bridge );
    if(m_dbg) m_dbg->setMaterialBridge( m_material_bridge );
}

void CRecorder::initEvent(OpticksEvent* evt)  // called by CG4::initEvent
{
    assert(evt);
    m_writer->initEvent(evt);
    evt->setNote( m_recpoi ? "recpoi" : "recstp" );
}

void CRecorder::posttrack() // invoked from CTrackingAction::PostUserTrackingAction
{
    assert(!m_live);

    if(m_ctx._dbgrec) LOG(info) << "CRecorder::posttrack" ; 

    if(m_recpoi)
    {
        posttrackWritePoints();  // experimental alt 
    }
    else
    {
        posttrackWriteSteps();
    } 


    if(m_dbg) m_dbg->posttrack(); 
}


/**
CRecorder::Record
===================

Not-zeroing m_slot for REJOINders 
----------------------------------

* see notes/issues/reemission_review.rst

Rejoining happens on output side not in the crec CStp list.

The rejoins of AB(actually RE) tracks with reborn secondaries 
are done by writing two (or more) sequences of track steps  
into the same record_id in the record buffer at the 
appropriate non-zeroed slot.

WAS a bit confused by this ...
 
This assumes that the REJOINing track will
be the one immediately after the original AB. 
By virtue of the Cerenkov/Scintillation process setting:

     SetTrackSecondariesFirst(true)
  
If not so, this will "join" unrelated tracks ?

**/

// invoked by CSteppingAction::setStep
// stage is set by CG4Ctx::setStepOptical from CSteppingAction::setStep
#ifdef USE_CUSTOM_BOUNDARY
bool CRecorder::Record(DsG4OpBoundaryProcessStatus boundary_status)
#else
bool CRecorder::Record(G4OpBoundaryProcessStatus boundary_status)
#endif
{    
    m_state._step_action = 0 ; 

    assert(!m_live);


    if(m_ctx._stage == CStage::START)
    { 
        zeroPhoton();
    }
    else if(m_ctx._stage == CStage::REJOIN )
    {
        m_crec->clear(); // NB Not-zeroing m_slot for REJOINders, see above note
    }
    else if(m_ctx._stage == CStage::RECOLL )
    {
        m_state._decrement_request = 0 ;  
    } 

    bool done = m_crec->add(boundary_status) ;

    if(m_ctx._dbgrec)
        LOG(info) << "crec.add "
                  << "[" 
                  << std::setw(2) << m_crec->getNumStp()
                  << "]"
                  << std::setw(10) << CStage::Label(m_ctx._stage)
                  << " " << m_ctx.desc_step() 
                  << " " << ( done ? "DONE" : "-" )
                  ; 

    return done ; 
}


void CRecorder::zeroPhoton()
{  
    const G4StepPoint* pre = m_ctx._step->GetPreStepPoint() ;
    const G4ThreeVector& pos = pre->GetPosition();
    m_crec->setOrigin(pos);   // hmm maybe in CG4Ctx already ?
    m_crec->clear();

    m_photon.clear();
    m_state.clear();

    if(m_dbg) m_dbg->Clear();
}



void CRecorder::posttrackWritePoints()
{
#ifdef USE_CUSTOM_BOUNDARY
    DsG4OpBoundaryProcessStatus boundary_status = Undefined ;
#else
    G4OpBoundaryProcessStatus boundary_status = Undefined ;
#endif
 
    unsigned num = m_crec->getNumPoi(); 
    for(unsigned i=0 ; i < num ; i++)
    {
        m_state._step_action = 0 ; 
        CPoi* poi  = m_crec->getPoi(i);

        const G4StepPoint* point = poi->getPoint();
        unsigned flag = poi->getFlag(); 
        unsigned material = poi->getMaterial() ; 
        boundary_status = poi->getBoundaryStatus() ; 
        
        RecordStepPoint( point, flag, material, boundary_status, NULL );

        //if(done) assert( i == num - 1 ) ; 
    } 
}


void CRecorder::posttrackWriteSteps()
{
   // CRecorder::posttrackWriteSteps is invoked from CRecorder::posttrack, 
   // once for the primary photon track and then for 0 or more reemtracks
   // via the record_id the info is written onto the correct place 
   // in the photon record buffer
    assert(!m_live) ;

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

    unsigned num = m_crec->getNumStp(); 

    bool limited = m_crec->is_step_limited() ; 

    if(m_ctx._dbgrec)
    {
        LOG(info) << "CRecorder::posttrackWriteSteps"
                  << " [--dbgrec] "
                  << " num " << num
                  << " m_slot " << m_state._slot
                  << " is_step_limited " << ( limited ? "Y" : "N" )
                   ;

        std::cout << "CRecorder::posttrackWriteSteps stages:"  ;
        for(unsigned i=0 ; i < num ; i++) std::cout << CStage::Label(m_crec->getStp(i)->getStage()) << " " ; 
        std::cout << std::endl ;  
    }


    unsigned i = 0 ;  
    for(i=0 ; i < num ; i++)
    {
        m_state._step_action = 0 ; 

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

        unsigned postFlag = OpStatus::OpPointFlag(post, boundary_status, postStage);

        bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS )) != 0 ;

        bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

        bool postSkip = boundary_status == StepTooSmall && !lastPost  ;  

        bool matSwap = next_boundary_status == StepTooSmall ; 

        // see notes/issues/geant4_opticks_integration/tconcentric_post_recording_has_seqmat_zeros.rst

        if(lastPost)      m_state._step_action |= CAction::LAST_POST ; 
        if(surfaceAbsorb) m_state._step_action |= CAction::SURF_ABS ;  
        if(postSkip)      m_state._step_action |= CAction::POST_SKIP ; 
        if(matSwap)       m_state._step_action |= CAction::MAT_SWAP ; 

        switch(stage)
        {
            case CStage::START:  m_state._step_action |= CAction::STEP_START    ; break ; 
            case CStage::REJOIN: m_state._step_action |= CAction::STEP_REJOIN   ; break ; 
            case CStage::RECOLL: m_state._step_action |= CAction::STEP_RECOLL   ; break ;
            case CStage::COLLECT:                               ; break ; 
            case CStage::UNKNOWN:assert(0)                      ; break ; 
        } 

        bool first = m_state._slot == 0 && stage == CStage::START ;

        unsigned u_premat  = matSwap ? postmat : premat ;
        unsigned u_postmat = ( matSwap || postmat == 0 )  ? premat  : postmat ;

        if(first)            u_premat = premat ;   
        // dont allow any matswap for 1st material : see notes/issues/tboolean-box-okg4-seqmat-mismatch.rst
        if(surfaceAbsorb)    u_postmat = postmat ; 


        if(stage == CStage::REJOIN) 
        {
             m_state.decrementSlot();   // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
        }

       // as clearStp for each track, REJOIN will always be i=0

        unsigned preFlag = first ? m_ctx._gen : OpStatus::OpPointFlag(pre,  prior_boundary_status, stage) ;

        if(i == 0)
        {

            m_state._step_action |= CAction::PRE_SAVE ; 

            done = RecordStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE );   

            if(done) m_state._step_action |= CAction::PRE_DONE ; 

            if(!done)
            {
                 done = RecordStepPoint( post, postFlag, u_postmat, boundary_status,       POST );  

                 m_state._step_action |= CAction::POST_SAVE ; 

                 if(done) m_state._step_action |= CAction::POST_DONE ; 
            }
        }
        else
        {
            if(!postSkip && !done)
            {
                m_state._step_action |= CAction::POST_SAVE ; 

                done = RecordStepPoint( post, postFlag, u_postmat, boundary_status, POST );

                if(done) m_state._step_action |= CAction::POST_DONE ; 
            }
        }

        // huh: changing the inputs (for dumping presumably) ??? nevertheless confusing 
        // ... but are using next step lookahead, 
        stp->setMat(  u_premat, u_postmat );
        stp->setFlag( preFlag,  postFlag );
        stp->setAction( m_state._step_action );


        if(m_ctx._dbgrec)
            LOG(info) << "[--dbgrec] posttrackWriteSteps " 
                      << "[" << std::setw(2) << i << "]"
                      << " action " << getStepActionString()
                      ;


        if(done) break ; 


    }   // stp loop


    if(!done)
    {
        m_not_done_count++ ; 
        LOG(fatal) << "posttrackWriteSteps  not-done " 
                   << m_not_done_count
                   << " photon " << m_photon.desc()
                   << " action " << getStepActionString()
                   << " i " << i 
                   << " num " << num 
                   ; 
    } 

}



#ifdef USE_CUSTOM_BOUNDARY
bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* )
#else
bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* )
#endif
{

    if(flag == 0)
    {
        if(!(boundary_status == SameMaterial || boundary_status == Undefined))
            LOG(warning) << " boundary_status not handled : " << OpStatus::OpBoundaryAbbrevString(boundary_status) ; 
    }

    return m_writer->writeStepPoint( point, flag, material );
}


/*

        if(m_ctx._debug || m_ctx._other)
        {
            LOG(info)
                  << ( m_state._topslot_rewrite > 1 ? CAction::HARD_TRUNCATE_ : CAction::TOPSLOT_REWRITE_ )
                  << " topslot_rewrite " << m_state._topslot_rewrite
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

*/







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
       << " slt " << std::setw(4) << m_state._slot
       << " pre " << std::setw(7) << CStep::PreGlobalTime(m_ctx._step)
       << " pst " << std::setw(7) << CStep::PostGlobalTime(m_ctx._step)
       << ( m_dynamic ? " DYNAMIC " : " STATIC " )
       ;

   return ss.str();
}

