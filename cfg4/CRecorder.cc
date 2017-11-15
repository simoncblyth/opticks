#include <sstream>
#include "CFG4_BODY.hh"
#include "CBoundaryProcess.hh"

// brap-
#include "BStr.hh"
#include "BBit.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksQuadrant.h"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"


// g4-
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

// cfg4-
#include "CG4.hh"
#include "OpStatus.hh"
#include "CRecorder.h"
#include "CPropLib.hh"
#include "Format.hh"
#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "CWriter.hh"
#include "CRec.hh"
#include "CStp.hh"
#include "State.hh"
#include "CStep.hh"
#include "CAction.hh"
#include "CRecorder.hh"

#include "PLOG.hh"



const char* CRecorder::PRE  = "PRE" ; 
const char* CRecorder::POST = "POST" ; 

/**
CRecorder
==========

Canonical instance is ctor resident of CG4 

**/

CRecorder::CRecorder(CG4* g4, CGeometry* geometry, bool dynamic) 
   :
   m_g4(g4),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),
   m_dbgseqhis(m_ok->getDbgSeqhis()),
   m_dbgseqmat(m_ok->getDbgSeqmat()),
   m_dbgflags(m_ok->hasOpt("dbgflags")),
   m_crec(new CRec(m_g4, geometry, dynamic)),
   m_evt(NULL),
   m_geometry(geometry),
   m_material_bridge(NULL),
   m_dynamic(dynamic),
   m_live(m_ok->hasOpt("liverecorder")),
   m_writer(new CWriter(g4, m_dynamic)),

   m_verbosity(m_ok->hasOpt("steppingdbg") ? 10 : 0),

   m_boundary_status(Undefined),
   m_prior_boundary_status(Undefined),

   m_premat(0),
   m_prior_premat(0),

   m_postmat(0),
   m_prior_postmat(0),

   m_seqhis_select(0),
   m_seqmat_select(0),
   m_slot(0),
   m_decrement_request(0),
   m_decrement_denied(0),
   m_record_truncate(false),
   m_bounce_truncate(false),
   m_topslot_rewrite(0),
   m_badflag(0),
   m_step_action(0)
{
   
}



void CRecorder::postinitialize()
{
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);
}

unsigned int CRecorder::getVerbosity()
{
    return m_verbosity ; 
}
bool CRecorder::isHistorySelected()
{
   return m_seqhis_select == m_photon._seqhis ; 
}
bool CRecorder::isMaterialSelected()
{
   return m_seqmat_select == m_photon._seqmat ; 
}
bool CRecorder::isSelected()
{
   return isHistorySelected() || isMaterialSelected() ;
}

unsigned long long CRecorder::getSeqHis()
{
    return m_photon._seqhis ; 
}
unsigned long long CRecorder::getSeqMat()
{
    return m_photon._seqmat ; 
}



std::string CRecorder::description()
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


std::string CRecorder::desc() const 
{
    std::stringstream ss ; 

    ss << "CRecorder"
       << " live " << m_live 
       << " dynamic " << m_dynamic
       << " record_max " << m_ctx._record_max
       << " bounce_max " << m_ctx._bounce_max
       ; 

    return ss.str();
}

void CRecorder::RecordBeginOfRun(const G4Run*)
{
}

void CRecorder::RecordEndOfRun(const G4Run*)
{
}

void CRecorder::initEvent(OpticksEvent* evt)  // called by CG4::initEvent
{
    m_writer->initEvent(evt);


}






unsigned CRecorder::getSlot()
{
    return m_slot ; 
}
void CRecorder::setSlot(unsigned slot) // needed for reemission continuation
{
    m_slot = slot ; 
}

void CRecorder::startPhoton()
{
   // invoked from CRecorder::Record when stage = CStage::START
   // the start stage is set for a new non-rejoing optical track by   CSteppingAction::UserSteppingActionOptical

    if(m_ctx._dbgrec)
    {
        LOG(info) << "[--dbgrec] " 
                  << " m_slot " << m_slot 
                  ;
    }

    const G4StepPoint* pre = m_ctx._step->GetPreStepPoint() ;
    const G4ThreeVector& pos = pre->GetPosition();

    m_crec->startPhoton(pos);   // clears CStp vector

    // description of photon history 


    m_boundary_status = Undefined ; 
    m_prior_boundary_status = Undefined ; 

    m_premat = 0 ; 
    m_prior_premat = 0 ; 

    m_postmat = 0 ; 
    m_prior_postmat = 0 ; 


    m_photon.clear();

    m_seqhis_select = 0x8bd ;

    m_slot = 0 ; 
    m_decrement_request = 0 ; 
    m_decrement_denied = 0 ; 
    m_record_truncate = false ; 
    m_bounce_truncate = false ; 
    m_topslot_rewrite = 0 ; 

    m_badflag = 0 ; 


    if(m_ctx._debug || m_ctx._other) Clear();
}

void CRecorder::decrementSlot()
{
    m_decrement_request += 1 ; 

    //if(m_slot == 0 || m_bounce_truncate || m_record_truncate )
    // with the TOPSLOT_REWRITE dont want to deny decrement
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


// invoked by CSteppingAction::setStep
#ifdef USE_CUSTOM_BOUNDARY
bool CRecorder::Record(DsG4OpBoundaryProcessStatus boundary_status)
#else
bool CRecorder::Record(G4OpBoundaryProcessStatus boundary_status)
#endif
{
    
    //bool recording = unsigned(m_ctx._record_id) < m_record_max ||  m_dynamic ;  // record_max is a photon level fit-in-buffer thing
    //if(!recording) 
    //{
    //    assert(0);
    //    return false ;
    //}

    m_step_action = 0 ; 

    if(m_ctx._dbgrec)
    LOG(trace) << "CRecorder::Record"
              << " step_id " << m_ctx._step_id
              << " record_id " << m_ctx._record_id
              << " stage " << CStage::Label(m_ctx._stage)
              ;

    if(m_ctx._stage == CStage::START)
    { 
        startPhoton();       // MUST be invoked prior to setBoundaryStatus, resetting photon history state 
        RecordQuadrant(m_photon._c4);
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
         m_crec->add(m_ctx._step, m_ctx._step_id, m_boundary_status, m_ctx._stage );

         if(m_ctx._dbgrec)
             LOG(info) << "[--dbgrec]" 
                       << " crec.add NumStps " << m_crec->getNumStps() 
                       ; 



    }

    return done ; 
}


void CRecorder::CannedWriteSteps()
{
   // CRecorder::CannedWriteSteps is invoked from CRecorder::posttrack, 
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
        LOG(info) << "CRecorder::CannedWriteSteps"
                  << " [--dbgrec] "
                  << " num " << num
                  << " m_slot " << m_slot
                   ;

        std::cout << "CRecorder::CannedWriteSteps stages:"  ;
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


        unsigned u_premat  = matSwap ? postmat : premat ;
        unsigned u_postmat = ( matSwap || postmat == 0 )  ? premat  : postmat ;

        if(surfaceAbsorb) u_postmat = postmat ; 

        bool first = m_slot == 0 && stage == CStage::START ;

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

        assert( m_photon._seqhis ) ;  // not tripped
        assert( m_photon._seqmat ) ; 
        assert( m_photon._mskhis ) ; 

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


    if(num > 0 )
    {
    assert( m_photon._seqhis ) ; 
    assert( m_photon._seqmat ) ; 
    assert( m_photon._mskhis ) ; 
    }



}


/**
   Consider 
       TO RE BT BT BT BT SA

   Live mode:
       write pre until last when write pre,post 

   Canned mode:
        For first write pre,post then write post

   Rejoins are not known until another track comes along 
   that lines up with former ending in AB. 
**/



void CRecorder::posttrack()
{
    // invoked from CTrackingAction::PostUserTrackingAction

    if(m_ctx._dbgrec) LOG(info) << "CRecorder::posttrack" ; 

    if(!m_live)
    { 
        CannedWriteSteps();
        //assert( m_photon._seqhis ); // trips
        //assert( m_photon._seqmat );
    }


    //assert( m_photon._seqhis );
    //assert( m_photon._seqmat );


    if(m_dbgflags && m_badflag > 0) addDebugPhoton(m_ctx._record_id);  

    bool debug_seqhis = m_dbgseqhis == m_photon._seqhis ; 
    bool debug_seqmat = m_dbgseqmat == m_photon._seqmat ; 

    bool dump_ = m_verbosity > 0 || debug_seqhis || debug_seqmat || m_ctx._other || m_ctx._debug || (m_dbgflags && m_badflag > 0 ) ;

    if(m_badflag > 0) dump_ = true ; 


    //if(dump_) dump("CRecorder::posttrack");
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
       m_badflag += 1 ; 
       m_step_action |= CAction::ZERO_FLAG ; 

       if(!(boundary_status == SameMaterial || boundary_status == Undefined))
            LOG(warning) << " boundary_status not handled : " << OpBoundaryAbbrevString(boundary_status) ; 
    }


    m_photon.add(slot, flag, material);

    if(m_record_truncate && m_photon.is_rewrite_slot() )  // try to overwrite top slot 
    {
        m_topslot_rewrite += 1 ; 
        if(m_ctx._debug || m_ctx._other)
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

        // allowing a single AB->RE rewrite is closer to Opticks
        if(m_topslot_rewrite == 1 && flag == BULK_REEMIT && m_photon._flag_prior  == BULK_ABSORB)
        {
            m_step_action |= CAction::TOPSLOT_REWRITE ; 
        }
        else
        {
            m_step_action |= CAction::HARD_TRUNCATE ; 
            assert(0);
            return true ; 
        }
    }

   if(flag == BULK_REEMIT) m_photon._mskhis = m_photon._mskhis & (~BULK_ABSORB)  ;


    unsigned target_record_id = m_dynamic ? 0 : m_ctx._record_id ; 


    m_writer->writeStepPoint(target_record_id, slot, point, flag, material, label);


    if(m_ctx._debug || m_ctx._other) Collect(point, boundary_status, m_photon );

    m_slot += 1 ;    // m_slot is incremented regardless of truncation, only local *slot* is constrained to recording range

    //LOG(info) << " inc slot " << m_slot ; 


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




    /*
    std::cout << "RecordStepPoint.slot "
              << std::setw(4) << slot << " "
              << std::setw(16) << std::hex << m_seqhis << std::dec 
              << " " << OpticksFlags::FlagSequence(m_seqhis, true)
              << std::endl 
              ;
    */


    //  Decrementing m_slot and running again will not scrub the AB from the mask
    //  so need to scrub the AB (BULK_ABSORB) when a RE (BULK_REEMIT) from rejoining
    //  occurs. 
    //
    //  Thus should always be correct as AB is a terminating flag, 
    //  so any REJOINed photon will have an AB in the mask
    //  that needs to be a RE instead.
    //
    //  What about SA/SD ... those should never REjoin ?




std::string CRecorder::getStepActionString()
{
    return CAction::Action(m_step_action) ;
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








#ifdef USE_CUSTOM_BOUNDARY
void CRecorder::Collect(const G4StepPoint* point, DsG4OpBoundaryProcessStatus boundary_status, const CPhoton& photon )
#else
void CRecorder::Collect(const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, const CPhoton& photon )
#endif
{

    double time = point->GetGlobalTime();

    assert(m_ctx._debug || m_ctx._other);

    m_points.push_back(new G4StepPoint(*point));
    m_flags.push_back(photon._flag);
    m_materials.push_back(photon._material);
    m_bndstats.push_back(boundary_status);  // will duplicate the status for the last step
    m_mskhis_dbg.push_back(photon._mskhis);
    m_seqhis_dbg.push_back(photon._seqhis);
    m_seqmat_dbg.push_back(photon._seqmat);
    m_times.push_back(time);
}


void CRecorder::Clear()
{
    assert(m_ctx._debug || m_ctx._other);
    for(unsigned int i=0 ; i < m_points.size() ; i++) delete m_points[i] ;
    m_points.clear();
    m_flags.clear();
    m_materials.clear();
    m_bndstats.clear();
    m_seqhis_dbg.clear();
    m_seqmat_dbg.clear();
    m_mskhis_dbg.clear();
    m_times.clear();
}



#ifdef USE_CUSTOM_BOUNDARY
DsG4OpBoundaryProcessStatus CRecorder::getBoundaryStatus()
#else
G4OpBoundaryProcessStatus CRecorder::getBoundaryStatus()
#endif
{
   return m_boundary_status ; 
}





void CRecorder::RecordQuadrant(uifchar4& c4)
{
    const G4StepPoint* pre  = m_ctx._step->GetPreStepPoint() ; 
    const G4ThreeVector& pos = pre->GetPosition();

    // initial quadrant 
    c4.uchar_.x = 
                  (  pos.x() > 0.f ? unsigned(QX) : 0u ) 
                   |   
                  (  pos.y() > 0.f ? unsigned(QY) : 0u ) 
                   |   
                  (  pos.z() > 0.f ? unsigned(QZ) : 0u )
                  ;   

    c4.uchar_.y = 2u ; 
    c4.uchar_.z = 3u ; 
    c4.uchar_.w = 4u ; 
}





bool CRecorder::hasIssue()
{
    unsigned int npoints = m_points.size() ;
    assert(m_flags.size() == npoints);
    assert(m_materials.size() == npoints);
    assert(m_bndstats.size() == npoints);

    bool issue = false ; 
    for(unsigned int i=0 ; i < npoints ; i++) 
    {
       if(m_flags[i] == 0 || m_flags[i] == NAN_ABORT) issue = true ; 
    }
    return issue ; 
}

void CRecorder::Summary(const char* msg)
{
    LOG(info) <<  msg
              << " event_id " << m_ctx._event_id 
              << " photon_id " << m_ctx._photon_id 
              << " record_id " << m_ctx._record_id 
              << " step_id " << m_ctx._step_id 
              << " m_slot " << m_slot 
              ;
}

void CRecorder::addSeqhisMismatch(unsigned long long rdr, unsigned long long rec)
{
    m_seqhis_mismatch.push_back(std::pair<unsigned long long, unsigned long long>(rdr, rec));
}
void CRecorder::addSeqmatMismatch(unsigned long long rdr, unsigned long long rec)
{
    m_seqmat_mismatch.push_back(std::pair<unsigned long long, unsigned long long>(rdr, rec));
}
void CRecorder::addDebugPhoton(int record_id)
{
    m_debug_photon.push_back(record_id);
}



void CRecorder::dump(const char* msg)
{
    LOG(info) << msg ; 
    dump_brief("CRecorder::dump_brief");

/*
    dump_brief("CRecorder::dump_brief");
    if(m_debug || m_other ) 
    {
        dump_sequence("CRecorder::dump_sequence");
        dump_points("CRecorder::dump_points");
    }
    m_crec->dump("CRec::dump");
*/
}

void CRecorder::dump_brief(const char* msg)
{
    LOG(info) << msg 
              << " m_ctx._record_id " << std::setw(8) << m_ctx._record_id 
              << " m_badflag " << std::setw(5) << m_badflag 
              << (m_ctx._debug ? " --dindex " : "" )
              << (m_ctx._other ? " --oindex " : "" )
              << (m_dbgseqhis == m_photon._seqhis ? " --dbgseqhis " : "" )
              << (m_dbgseqmat == m_photon._seqmat ? " --dbgseqmat " : "" )
              << " sas: " << getStepActionString()
              ;
    LOG(info) 
              << " seqhis " << std::setw(16) << std::hex << m_photon._seqhis << std::dec 
              << "    " << OpticksFlags::FlagSequence(m_photon._seqhis, true) 
              ;

    LOG(info) 
              << " mskhis " << std::setw(16) << std::hex << m_photon._mskhis << std::dec 
              << "    " << OpticksFlags::FlagMask(m_photon._mskhis, true) 
              ;

    LOG(info) 
              << " seqmat " << std::setw(16) << std::hex << m_photon._seqmat << std::dec 
              << "    " << m_material_bridge->MaterialSequence(m_photon._seqmat) 
              ;
}

void CRecorder::dump_sequence(const char* msg)
{
    assert(m_ctx._debug || m_ctx._other) ; // requires premeditation to collect the info
    LOG(info) << msg ; 
    unsigned npoints = m_points.size() ;
    for(unsigned int i=0 ; i<npoints ; i++) 
        std::cout << std::setw(4) << i << " "
                  << std::setw(16) << std::hex << m_seqhis_dbg[i] << std::dec 
                  << " " << OpticksFlags::FlagSequence(m_seqhis_dbg[i], true)
                  << std::endl 
                  ;

    for(unsigned int i=0 ; i<npoints ; i++) 
        std::cout << std::setw(4) << i << " "
                  << std::setw(16) << std::hex << m_mskhis_dbg[i] << std::dec 
                  << " " << OpticksFlags::FlagMask(m_mskhis_dbg[i], true)
                  << std::endl 
                  ;

    for(unsigned int i=0 ; i<npoints ; i++) 
        std::cout << std::setw(4) << i << " "
                  << std::setw(16) << std::hex << m_seqmat_dbg[i] << std::dec 
                  << " " << m_material_bridge->MaterialSequence(m_seqmat_dbg[i]) 
                  << std::endl 
                  ;
}

void CRecorder::dump_points(const char* msg)
{
    assert(m_ctx._debug || m_ctx._other) ; // requires premeditation to collect the info
    LOG(info) << msg ; 
    G4ThreeVector origin ;
    unsigned npoints = m_points.size() ;
    if(npoints > 0) origin = m_points[0]->GetPosition();

    for(unsigned int i=0 ; i<npoints ; i++) 
    {
        unsigned mat = m_materials[i] ;
        const char* matname = ( mat == 0 ? "-" : m_material_bridge->getMaterialName(mat-1)  ) ;
        dump_point(origin, i, m_points[i], m_bndstats[i], m_flags[i], matname );
    }
}


#ifdef USE_CUSTOM_BOUNDARY
void CRecorder::dump_point(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, DsG4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname )
#else
void CRecorder::dump_point(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname )
#endif
{
    std::string bs = OpBoundaryAbbrevString(boundary_status) ;
    const char* flg = OpticksFlags::Abbrev(flag) ;
    std::cout << std::setw(3) << flg << std::setw(7) << index << " " << std::setw(18) << matname << " " << Format(point, origin, bs.c_str()) << std::endl ;
}



void CRecorder::dumpStepVelocity(const char* msg )
{
    // try to understand GlobalTime calc from G4Transportation::AlongStepDoIt by duping attempt
    // issue is what velocity it gets to use, and the updating of that 

    G4Track* track = m_ctx._step->GetTrack() ;
    G4double trackStepLength = track->GetStepLength();
    G4double trackGlobalTime = track->GetGlobalTime() ;
    G4double trackVelocity = track->GetVelocity() ;

    const G4StepPoint* pre  = m_ctx._step->GetPreStepPoint() ; 
    const G4StepPoint* post = m_ctx._step->GetPostStepPoint() ; 


    G4double preDeltaTime = 0.0 ; 
    G4double preVelocity = pre->GetVelocity();
    if ( preVelocity > 0.0 )  { preDeltaTime = trackStepLength/preVelocity; }

    G4double postDeltaTime = 0.0 ; 
    G4double postVelocity = post->GetVelocity();
    if ( postVelocity > 0.0 )  { postDeltaTime = trackStepLength/postVelocity; }


    LOG(info) << msg
              << " trackStepLength " << std::setw(10) << trackStepLength 
              << " trackGlobalTime " << std::setw(10) << trackGlobalTime
              << " trackVelocity " << std::setw(10) << trackVelocity
              << " preVelocity " << std::setw(10) << preVelocity
              << " postVelocity " << std::setw(10) << postVelocity
              << " preDeltaTime " << std::setw(10) << preDeltaTime
              << " postDeltaTime " << std::setw(10) << postDeltaTime
              ;

}



void CRecorder::report(const char* msg)
{
     LOG(info) << msg ;
     unsigned cut = 50 ; 

     typedef std::vector<std::pair<unsigned long long, unsigned long long> >  VUU ; 
   
     unsigned nhis = m_seqhis_mismatch.size() ;
     unsigned ihis(0); 
     LOG(info) << " seqhis_mismatch " << nhis ;
     for(VUU::const_iterator it=m_seqhis_mismatch.begin() ; it != m_seqhis_mismatch.end() ; it++)
     { 
          ihis++ ;
          if(ihis < cut || ihis > nhis - cut )
          {
              unsigned long long rdr = it->first ;
              unsigned long long rec = it->second ;
              std::cout 
                        << " ihis " << std::setw(10) << ihis
                        << " rdr " << std::setw(16) << std::hex << rdr << std::dec
                        << " rec " << std::setw(16) << std::hex << rec << std::dec
                    //    << " rdr " << std::setw(50) << OpticksFlags::FlagSequence(rdr)
                    //    << " rec " << std::setw(50) << OpticksFlags::FlagSequence(rec)
                        << std::endl ; 
          }
          else if(ihis == cut)
          {
                std::cout << " ... " << std::endl ; 
          }
     }

     unsigned nmat = m_seqmat_mismatch.size() ;
     unsigned imat(0); 
     LOG(info) << " seqmat_mismatch " << nmat ;
     for(VUU::const_iterator it=m_seqmat_mismatch.begin() ; it != m_seqmat_mismatch.end() ; it++)
     {
          imat++ ; 
          if(imat < cut || imat > nmat - cut)
          {
              unsigned long long rdr = it->first ;
              unsigned long long rec = it->second ;
              std::cout 
                        << " imat " << std::setw(10) << imat
                        << " rdr " << std::setw(16) << std::hex << rdr << std::dec
                        << " rec " << std::setw(16) << std::hex << rec << std::dec
                        << " rdr " << std::setw(50) << m_material_bridge->MaterialSequence(rdr)
                        << " rec " << std::setw(50) << m_material_bridge->MaterialSequence(rec)
                        << std::endl ; 
           } 
           else if(imat == cut)
           {
                std::cout << " ... " << std::endl ; 
           }
     }


     unsigned ndbg = m_debug_photon.size() ;
     LOG(info) << " debug_photon " << ndbg << " (photon_id) " ; 
     typedef std::vector<int> VI ; 
     if(ndbg < 100) 
     for(VI::const_iterator it=m_debug_photon.begin() ; it != m_debug_photon.end() ; it++) std::cout << std::setw(8) << *it << std::endl ; 

     LOG(info) << "TO DEBUG THESE USE:  --dindex=" << BStr::ijoin(m_debug_photon, ',') ;

}




