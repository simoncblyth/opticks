#include "Recorder.hh"
#include "Format.hh"

#include "G4RunManager.hh"
#include "G4Event.hh"

#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


#include "NPY.hpp"
#include "NLog.hpp"







void Recorder::init()
{
    // uncompressed initially
    m_recs = NPY<float>::make( m_record_max, m_steps_per_photon, 4, 4) ; 
    m_recs->zero();
}

void Recorder::save(const char* path)
{
    LOG(info) << "Recorder::save " << path  ;
    m_recs->save(path);
}

void Recorder::RecordBeginOfRun(const G4Run*)
{
}
void Recorder::RecordEndOfRun(const G4Run*)
{
}


void Recorder::RecordStep(const G4Step* step)
{
    unsigned int eid = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    G4Track* track = step->GetTrack();
    G4int tid = track->GetTrackID();
    G4int sid = track->GetCurrentStepNumber() ;
    assert(tid >= 1 && sid >= 1);   // one-based

    bool startEvent = eid != m_event_id ; 
     
    // make all Id zero-based
    setEventId(eid);
    setPhotonId(tid-1);  
    setStepId(sid-1);
    unsigned int record_id = getRecordId();

    //if(m_step_status != FresnelRefraction)
    //if(m_photon_id < 3) 
    //LOG(info) << Format(m_step_status) << Format(step) ; 

    //if(startEvent || m_photon_id % 1000 == 0)
    if(m_photon_id < 10) 
    LOG(info) << "Recorder::RecordStep" 
              << " event_id " << m_event_id 
              << " photon_id " << m_photon_id 
              << " step_id " << m_step_id 
              << " record_id " << record_id 
               ;


    //TODO: handle truncation in equivalent way to optixrap-
    if(record_id < m_record_max && m_step_id+1 < m_steps_per_photon )
    {     
        G4StepPoint* pre  = step->GetPreStepPoint() ; 
        G4StepPoint* post = step->GetPostStepPoint() ; 
        G4StepStatus postStatus = post->GetStepStatus()  ;

        bool lastStep = postStatus == fWorldBoundary ;   // TODO: other causes of lastStep: Absorption ...

        if(!lastStep)
        {
            RecordStepPoint(pre, record_id, m_step_id);
        }
        else
        {
            RecordStepPoint(pre,  record_id, m_step_id);
            RecordStepPoint(post, record_id, m_step_id+1);
        }
    }

}



void Recorder::RecordStepPoint(const G4StepPoint* point, unsigned int record_id, unsigned int step_id)
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

    m_recs->setQuad(record_id, step_id, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    m_recs->setQuad(record_id, step_id, 1, dir.x(), dir.y(), dir.z(), weight  );
    m_recs->setQuad(record_id, step_id, 2, pol.x(), pol.y(), pol.z(), wavelength/nm  );
}



