#include "Recorder.hh"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


#include "NPY.hpp"
#include "NLog.hpp"

void Recorder::init()
{
    // uncompressed initially
    m_recs = NPY<float>::make( m_photon_max, m_step_max, 4, 4) ; 
    m_recs->zero();
}

void Recorder::save(const char* path)
{
    LOG(info) << "Recorder::save " << path  ;
    m_recs->save(path);
}

void Recorder::RecordBeginOfRun(const G4Run*)
{
    LOG(info) << "Recorder::RecordBeginOfRun" ;
}
void Recorder::RecordEndOfRun(const G4Run*)
{
    LOG(info) << "Recorder::RecordEndOfRun" ;
}


void Recorder::RecordStep(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    G4int tid = track->GetTrackID();
    G4int sid = track->GetCurrentStepNumber() ;

    LOG(info) << "Recorder::RecordStep" 
              << " tid " << tid 
              << " sid " << sid 
               ;

    if(tid < m_photon_max && sid < m_step_max)
    {     
        assert(tid >= 1 && sid >= 1);

        G4StepPoint* point  = step->GetPreStepPoint() ;
        const G4ThreeVector& pos = point->GetPosition();
        const G4ThreeVector& dir = point->GetMomentumDirection();
        const G4ThreeVector& pol = point->GetPolarization();

        G4double time = point->GetGlobalTime();
        G4double energy = point->GetKineticEnergy();
        G4double wavelength = h_Planck*c_light/energy ;
        G4double weight = 1.0 ; 

        m_recs->setQuad(tid-1, sid-1, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
        m_recs->setQuad(tid-1, sid-1, 1, dir.x(), dir.y(), dir.z(), weight  );
        m_recs->setQuad(tid-1, sid-1, 2, pol.x(), pol.y(), pol.z(), wavelength/nm  );

    }


}



