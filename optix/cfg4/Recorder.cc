#include "Recorder.hh"
#include "Format.hh"
#include "OpStatus.hh"

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
    if(     strcmp(m_typ,"torch")==0)         m_gen = TORCH ;
    else if(strcmp(m_typ,"cerenkov")==0)      m_gen = CERENKOV ;
    else if(strcmp(m_typ,"scintillation")==0) m_gen = SCINTILLATION ;
    else
         assert(0);

    m_history = NPY<unsigned long long>::make( m_record_max, 1, 2) ; 
    m_history->zero();

    m_records = NPY<float>::make( m_record_max, m_steps_per_photon, 4, 4) ; 
    m_records->zero();
}

void Recorder::save()
{
    m_records->save("rx%s", m_typ, m_tag, m_det);
    m_history->save("ph%s", m_typ, m_tag, m_det);
}


#define RSAVE(flag, material, slot)  \
{    \
    unsigned int shift = slot*4 ; \
    unsigned long long his = ffs(flag) & 0xF ; \
    unsigned long long mat = material < 0xF ? material : 0xF ; \
    m_seqhis |= his << shift ; \
    m_seqmat |= mat << shift ; \
}   \



unsigned int Recorder::getPointFlag(G4StepPoint* point)
{
    G4StepStatus status = point->GetStepStatus()  ;

    const G4VProcess* process = point->GetProcessDefinedStep() ;
    const G4String& processName = process ? process->GetProcessName() : "NoProc" ; 

    bool transportation = strcmp(processName,"Transportation")==0 ;

    unsigned int flag(0);
    if(strcmp(processName,"OpAbsorption")==0 && status == fPostStepDoItProc )
    {
        flag = BULK_ABSORB ;
    }
    else if(transportation && status == fWorldBoundary )
    {
        flag = SURFACE_ABSORB ;  
        //kludge to match opticks use of perfect absorber at edge of world 
    }
    else if(transportation && status == fGeomBoundary )
    {
        flag = OpBoundaryFlag(m_boundary_status) ; 
    } 
    return flag ; 
}


void Recorder::RecordStep(const G4Step* step)
{
    unsigned int eid = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    bool startEvent = eid != m_event_id ; 
    setEventId(eid);

    G4Track* track = step->GetTrack();
    G4int tid = track->GetTrackID();

    G4int sid = track->GetCurrentStepNumber() ;
    assert(tid >= 1 && sid >= 1);   // one-based

    bool startPhoton = tid-1 != m_photon_id ; 
    setPhotonId(tid-1);   // zero-based
    setStepId(sid-1);

    unsigned int record_id = getRecordId();
    if(record_id >= m_record_max) return ; 

    G4StepPoint* pre  = step->GetPreStepPoint() ; 
    G4StepPoint* post = step->GetPostStepPoint() ; 

    // TODO: cache the relevant process objects, so can just compare pointers

    //  RecordStep is called for all G4Step
    //  each is comprised of pre and post points, so 
    //  see the same points twice : thus need to pick one to record
    //  except for the last 

    unsigned int preFlag  = m_step_id == 0 ? m_gen : getPointFlag(pre);
    unsigned int postFlag = getPointFlag(post) ;

    RecordStepPoint(pre, record_id, m_step_id, preFlag, false );
    if(postFlag & (BULK_ABSORB | SURFACE_ABSORB))
         RecordStepPoint(post, record_id, m_step_id+1, postFlag, true );


    if(m_photon_id < 10) 
    {
        G4StepStatus preStatus = pre->GetStepStatus()  ;
        G4StepStatus postStatus = post->GetStepStatus()  ;

        const G4VProcess* preProcess = pre->GetProcessDefinedStep() ;
        const G4String& preProcessName = preProcess ? preProcess->GetProcessName() : "NoPreProc" ; 

        const G4VProcess* postProcess = post->GetProcessDefinedStep() ;
        const G4String& postProcessName = postProcess ? postProcess->GetProcessName() : "NoPostProc" ; 

        if(startPhoton)
        LOG(info) 
              << "\n\n"
              << "Recorder::RecordStep" 
              << " photon_id " << m_photon_id
              ;

        LOG(info) 
              << " [" << m_step_id << "]"
              << " " << OpFlagString(preFlag) << "/" << OpFlagString(postFlag) 
              << " " << OpStepString(preStatus) << "/" << OpStepString(postStatus)
              << " " << OpBoundaryString(m_boundary_status) 
              << " " << preProcessName << "/" << postProcessName 
              << " eid " << m_event_id 
              << " pid " << m_photon_id 
              << " rid " << record_id 
              << "\n"
              << Format(step) ; 
              ;
    }

}



void Recorder::RecordStepPoint(const G4StepPoint* point, unsigned int record_id, unsigned int slot, unsigned int flag, bool last)
{
    unsigned int slot_offset =  slot < m_steps_per_photon  ? slot : m_steps_per_photon - 1 ;

    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

    m_records->setQuad(record_id, slot_offset, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    m_records->setQuad(record_id, slot_offset, 1, dir.x(), dir.y(), dir.z(), weight  );
    m_records->setQuad(record_id, slot_offset, 2, pol.x(), pol.y(), pol.z(), wavelength/nm  );


    unsigned int material = 0 ; 

    RSAVE(flag,material,slot_offset)

    if(last)
    {
        unsigned long long* history = m_history->getValues() + 2*record_id ;
        *(history+0) = m_seqhis ; 
        *(history+1) = m_seqmat ; 
    }
}

