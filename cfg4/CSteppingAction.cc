// g4-

#include "CFG4_PUSH.hh"

#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "DsG4CompositeTrackInfo.h"
#include "DsPhotonTrackInfo.h"


#include "BStr.hh"


// cg4-
#include "CBoundaryProcess.hh"

#include "CStage.hh"
#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "CRecorder.hh"
#include "Rec.hh"
#include "State.hh"
#include "Format.hh"
#include "CPropLib.hh"
#include "CStepRec.hh"
#include "CSteppingAction.hh"
#include "CTrack.hh"
#include "CG4.hh"

#include "CFG4_POP.hh"


// optickscore-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

// npy-
#include "PLOG.hh"


#ifdef USE_CUSTOM_BOUNDARY
DsG4OpBoundaryProcessStatus CSteppingAction::GetOpBoundaryProcessStatus()
{
    DsG4OpBoundaryProcessStatus status = Undefined;
#else
G4OpBoundaryProcessStatus CSteppingAction::GetOpBoundaryProcessStatus()
{
    G4OpBoundaryProcessStatus status = Undefined;
#endif
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    if(mgr) 
    {
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcess* opProc = NULL ;  
#else
        G4OpBoundaryProcess* opProc = NULL ;  
#endif
        G4int npmax = mgr->GetPostStepProcessVector()->entries();
        G4ProcessVector* pv = mgr->GetPostStepProcessVector(typeDoIt);
        for (G4int i=0; i<npmax; i++) 
        {
            G4VProcess* proc = (*pv)[i];
#ifdef USE_CUSTOM_BOUNDARY
            opProc = dynamic_cast<DsG4OpBoundaryProcess*>(proc);
#else
            opProc = dynamic_cast<G4OpBoundaryProcess*>(proc);
#endif
            if (opProc) 
            { 
                status = opProc->GetStatus(); 
                break;
            }
        }
    }
    return status ; 
}


/**
CSteppingAction
=================

Canonical instance (m_sa) is ctor resident of CG4 

**/

CSteppingAction::CSteppingAction(CG4* g4, bool dynamic)
   : 
   G4UserSteppingAction(),
   m_g4(g4),
   m_ok(g4->getOpticks()),
   m_dbgseqhis(m_ok->getDbgSeqhis()),
   m_dbgseqmat(m_ok->getDbgSeqmat()),
   m_dynamic(dynamic),
   m_geometry(g4->getGeometry()),
   m_material_bridge(NULL),
   m_clib(g4->getPropLib()),
   m_recorder(g4->getRecorder()),
   m_rec(g4->getRec()),
   m_steprec(g4->getStepRec()),
   m_verbosity(m_recorder->getVerbosity()),
   m_event_total(0),
   m_track_total(0),
   m_step_total(0),
   m_event_track_count(0),
   m_track_step_count(0),
   m_steprec_store_count(0),
   m_event(NULL),
   m_track(NULL),
   m_step(NULL),
   m_startEvent(false),
   m_startTrack(false),
   m_dindexDebug(false),
   m_event_id(-1),
   m_track_id(-1),
   m_parent_id(-1),
   m_step_id(-1),
   m_primary_id(-1),
   m_track_status(fAlive),
   m_particle(NULL),
   m_optical(false),
   m_pdg_encoding(0)
{ 
}


void CSteppingAction::postinitialize()
{
   // called from CG4::postinitialize
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);
}

CSteppingAction::~CSteppingAction()
{ 
}

const unsigned long long CSteppingAction::SEQHIS_TO_SA = 0x8dull ;     // Torch,SurfaceAbsorb
const unsigned long long CSteppingAction::SEQMAT_MO_PY_BK = 0x5e4ull ; // MineralOil,Pyrex,Bakelite?




void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent() ;

    int event_id = event->GetEventID() ;

    int track_id = track->GetTrackID() - 1 ;
    int parent_id = track->GetParentID() - 1 ;
    int step_id  = track->GetCurrentStepNumber() - 1 ;

    m_startEvent = m_event_id != event_id ; 
    m_startTrack = m_track_id != track_id || m_startEvent ; 

    if(m_startEvent) setEvent(event, event_id) ; 
    if(m_startTrack) setTrack(track, track_id, parent_id);

    bool done = setStep(step, step_id);

    if(done)
    { 
        track->SetTrackStatus(fStopAndKill);
        // stops tracking when reach truncation as well as absorption
    }
}


void CSteppingAction::setEvent(const G4Event* event, int event_id)
{
    LOG(info) << "CSA (startEvent)"
              << " event_id " << event_id
              << " event_total " <<  m_event_total
              ; 

    m_event = event ; 
    m_event_id = event_id ; 

    m_event_total += 1 ; 
    m_event_track_count = 0 ; 
}

void CSteppingAction::setTrack(const G4Track* track, int track_id, int parent_id)
{
    m_track = track ; 
    m_track_id = track_id ; 
    m_parent_id = parent_id ;
    m_track_status = track->GetTrackStatus(); 

    m_particle = track->GetDefinition();
    m_optical = m_particle == G4OpticalPhoton::OpticalPhotonDefinition() ;
    m_pdg_encoding = m_particle->GetPDGEncoding();

    m_event_track_count += 1 ; 
    m_track_total += 1 ; 

    m_track_step_count = 0 ; 
    m_rejoin_count = 0 ; 

    if(m_optical)
        LOG(trace) << "CSteppingAction::setTrack(optical)"
                  << " track_id " << m_track_id
                  << " parent_id " << m_parent_id
                  ;

}


bool CSteppingAction::setStep(const G4Step* step, int step_id)
{
    bool done = false ; 

    LOG(trace) << "CSteppingAction::setStep" 
               << " step_total " << m_step_total
               ; 

    m_step = step ; 
    m_step_id = step_id ; 

    m_track_step_count += 1 ; 
    m_step_total += 1 ; 

    if(m_optical)
    {
        done = UserSteppingActionOptical(step);
    }
    else
    {
        m_steprec->collectStep(step, step_id);

        if(m_track_status == fStopAndKill)
        {
            done = true ;  
            m_steprec->storeStepsCollected(m_event_id, m_track_id, m_pdg_encoding);
            m_steprec_store_count = m_steprec->getStoreCount(); 
        }
    }

    LOG(debug) << "    (step)"
              << " event_id " << m_event_id
              << " track_id " << m_track_id
              << " track_step_count " << m_track_step_count
              << " step_id " << m_step_id
              << " trackStatus " << CTrack::TrackStatusString(m_track_status)
              ;

   if(m_step_total % 10000 == 0) 
       LOG(debug) << "CSA (totals%10k)"
                 << " track_total " <<  m_track_total
                 << " step_total " <<  m_step_total
                 ;

    return done ;
}



int CSteppingAction::getPrimaryPhotonID()
{
    int primary_id = -2 ; 
    DsG4CompositeTrackInfo* cti = dynamic_cast<DsG4CompositeTrackInfo*>(m_track->GetUserInformation());
    if(cti)
    {
        DsPhotonTrackInfo* pti = dynamic_cast<DsPhotonTrackInfo*>(cti->GetPhotonTrackInfo());
        if(pti)
        {
            primary_id = pti->GetPrimaryPhotonID() ; 
        }
    }
    return primary_id ; 
}


bool CSteppingAction::UserSteppingActionOptical(const G4Step* step)
{
    bool done = false ; 
    int photon_id = m_track_id ;
    int parent_id = m_parent_id ;
    int step_id  = m_step_id ;

    assert( photon_id >= 0 );
    assert( step_id   >= 0 );
    assert( parent_id >= -1 );  // parent_id is -1 for primary photon
  
    int primary_id = getPrimaryPhotonID() ;
    int last_photon_id = m_recorder->getPhotonId();  

    CStage::CStage_t stage = CStage::UNKNOWN ; 
    if( parent_id == -1 )  // primary photon
    {
        stage = photon_id != last_photon_id  ? CStage::START : CStage::COLLECT ;
    } 
    else if( primary_id >= 0)
    {
        photon_id = primary_id ;  // <-- tacking reem step recording onto primary record 
        stage = m_rejoin_count == 0  ? CStage::REJOIN : CStage::RECOLL ;   
        m_rejoin_count++ ; 
        // rejoin count is zeroed in setTrack, so each remission generation track will result in REJOIN 
    }
     

    if(stage == CStage::START && last_photon_id >= 0 )
    {
        //  backwards looking comparison of prior (potentially rejoined) photon
        compareRecords(last_photon_id);
    }

 
    m_recorder->setPhotonId(photon_id);   
    m_recorder->setEventId(m_event_id);

    m_dindexDebug = m_ok->isDbgPhoton(photon_id) ; // from option: --dindex=1,100,1000,10000 
    m_recorder->setDebug(m_dindexDebug);

    unsigned int record_id = m_recorder->defineRecordId();   //  m_photons_per_g4event*m_event_id + m_photon_id 
    unsigned int record_max = m_recorder->getRecordMax() ;
    bool recording = record_id < record_max ||  m_dynamic ; 

    if(recording)
    {
        m_recorder->setStep(step);
        m_recorder->setStage(stage);
        m_recorder->setStepId(m_step_id);
        m_recorder->setParentId(parent_id);   
        m_recorder->setRecordId(record_id);

        if(stage == CStage::START)
        { 
            m_rec->Clear();
            m_recorder->startPhoton();  // MUST be invoked from up here,  prior to setBoundaryStatus
            m_recorder->RecordQuadrant(step);
        }
        else if(stage == CStage::REJOIN )
        {
            // back up by one, to scrub the "AB" an refill with "RE"
            // hmm what about when already truncating ?
            m_recorder->decrementSlot();    // this allows rewriting 
            m_rec->notifyRejoin(); 
        }



        const G4StepPoint* pre  = step->GetPreStepPoint() ; 
        const G4StepPoint* post = step->GetPostStepPoint() ; 

        const G4Material* preMat  = pre->GetMaterial() ;
        const G4Material* postMat = post->GetMaterial() ;

        unsigned preMaterial = preMat ? m_material_bridge->getMaterialIndex(preMat) + 1 : 0 ;
        unsigned postMaterial = postMat ? m_material_bridge->getMaterialIndex(postMat) + 1 : 0 ;

#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#else
        G4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#endif

        m_recorder->setBoundaryStatus(boundary_status, preMaterial, postMaterial);

        done = m_recorder->RecordStep();   // done=true for *absorption* OR *truncation*

        m_rec->add(new State(step, boundary_status, preMaterial, postMaterial, stage, done));

        if(done)
        {
            // compareRecords();  MOVED TO BACKWARDS LOOKING APPROACH  
            // splitting of a photon by G4/NuWa-Detsim-scintillation (reproduced in Scintillation) into separate tracks at each reemission
            // throws spanner in the works for inplace sequence comparison 
            //
            //  to regain the capability need to identify the rejoin lifecycle,
            //  a subsequent G4Step can rejoin onto the prior one
            //
            //  when move to a new primary photon (assuming immediate secondaries) could 
            //  be the juncture to make the comparison

            m_recorder->RecordPhoton(step);
        } 
    }
    return done ; 
}



void CSteppingAction::addSeqhisMismatch(unsigned long long rdr, unsigned long long rec)
{
    m_seqhis_mismatch.push_back(std::pair<unsigned long long, unsigned long long>(rdr, rec));
}
void CSteppingAction::addSeqmatMismatch(unsigned long long rdr, unsigned long long rec)
{
    m_seqmat_mismatch.push_back(std::pair<unsigned long long, unsigned long long>(rdr, rec));
}
void CSteppingAction::addDebugPhoton(int photon_id)
{
    m_debug_photon.push_back(photon_id);
}


void CSteppingAction::report(const char* msg)
{
    LOG(info) << msg ;

    std::cout 
           << " event_total " <<  m_event_total << std::endl 
           << " track_total " <<  m_track_total << std::endl 
           << " step_total " <<  m_step_total << std::endl 
           ;

     typedef std::vector<std::pair<unsigned long long, unsigned long long> >  VUU ; 
    
     LOG(info) << " seqhis_mismatch " << m_seqhis_mismatch.size() ;

     for(VUU::const_iterator it=m_seqhis_mismatch.begin() ; it != m_seqhis_mismatch.end() ; it++)
     { 
          unsigned long long rdr = it->first ;
          unsigned long long rec = it->second ;
          std::cout 
                    << " rdr " << std::setw(16) << std::hex << rdr << std::dec
                    << " rec " << std::setw(16) << std::hex << rec << std::dec
                //    << " rdr " << std::setw(50) << OpticksFlags::FlagSequence(rdr)
                //    << " rec " << std::setw(50) << OpticksFlags::FlagSequence(rec)
                    << std::endl ; 
     }

     LOG(info) << " seqmat_mismatch " << m_seqmat_mismatch.size() ; 
     for(VUU::const_iterator it=m_seqmat_mismatch.begin() ; it != m_seqmat_mismatch.end() ; it++)
     {
          unsigned long long rdr = it->first ;
          unsigned long long rec = it->second ;
          std::cout 
                    << " rdr " << std::setw(16) << std::hex << rdr << std::dec
                    << " rec " << std::setw(16) << std::hex << rec << std::dec
                    << " rdr " << std::setw(50) << m_material_bridge->MaterialSequence(rdr)
                    << " rec " << std::setw(50) << m_material_bridge->MaterialSequence(rec)
                    << std::endl ; 
     }

     LOG(info) << " debug_photon " << m_debug_photon.size() << " (photon_id) " ; 
     typedef std::vector<int> VI ; 
     for(VI::const_iterator it=m_debug_photon.begin() ; it != m_debug_photon.end() ; it++)
     {
         std::cout << std::setw(8) << *it << std::endl ; 
     }

     LOG(info) << "TO DEBUG THESE USE:  --dindex=" << BStr::ijoin(m_debug_photon, ',') ;

}

int CSteppingAction::compareRecords(int photon_id)
{
    assert(photon_id >= 0 );

    LOG(info) << "CSteppingAction::compareRecords"
              << " photon_id " << photon_id 
              ;

    int rc = 0 ; 

    unsigned long long rdr_seqhis = m_recorder->getSeqHis() ;
    unsigned long long rdr_seqmat = m_recorder->getSeqMat() ;

    bool debug_seqhis = m_dbgseqhis == rdr_seqhis ; 
    bool debug_seqmat = m_dbgseqmat == rdr_seqmat ; 

    //bool debug = rdr_seqmat == SEQMAT_MO_PY_BK && m_verbosity > 0 ;
    bool debug = m_verbosity > 0 || debug_seqhis || debug_seqmat || m_dindexDebug ;

    m_rec->setDebug(debug);
    m_rec->sequence();

    unsigned long long rec_seqhis = m_rec->getSeqHis() ;
    unsigned long long rec_seqmat = m_rec->getSeqMat() ;

    bool same_seqhis = rec_seqhis == rdr_seqhis ; 
    bool same_seqmat = rec_seqmat == rdr_seqmat ; 

    //assert(same_seqhis);
    //assert(same_seqmat);


    if(!same_seqhis) rc += 1 ; 
    if(!same_seqmat) rc += 1 ; 

    if(!same_seqhis) addSeqhisMismatch(rec_seqhis, rdr_seqhis);
    if(!same_seqmat) addSeqmatMismatch(rec_seqmat, rdr_seqmat);


    if(m_verbosity > 0 || debug || !same_seqhis || !same_seqmat  )
    {
        if(!same_seqmat || !same_seqhis || debug )
        {
            std::cout << std::endl 
                      << std::endl
                      << "----CSteppingAction----" 
                      << ( !same_seqhis  ? " !same_seqhis " : "" )
                      << ( !same_seqmat  ? " !same_seqmat " : "" )
                      << ( debug  ? " debug " : "" )
                      << std::endl 
                       ; 

            m_recorder->Dump("CSteppingAction::UserSteppingAction (rdr-dump)DONE");
            m_rec->Dump(     "CSteppingAction::UserSteppingAction (rec-dump)DONE");
        }

        if(!same_seqhis)
        { 
             std::cout << "(rec)" << OpticksFlags::FlagSequence(rec_seqhis) << std::endl ;  
             std::cout << "(rdr)" << OpticksFlags::FlagSequence(rdr_seqhis) << std::endl ;  
        }

        if(!same_seqmat)
        { 
             std::cout << "(rec)" << m_material_bridge->MaterialSequence(rec_seqmat) << std::endl ;  
             std::cout << "(rdr)" << m_material_bridge->MaterialSequence(rdr_seqmat) << std::endl ;  
        }
    }

    if(rc > 0)
    {
        addDebugPhoton(photon_id);  
    }
    return rc ; 
}
