#include "Format.hh"
#include "OpStatus.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include <sstream>


std::string Format(const G4ThreeVector& vec, const char* msg, unsigned int fwid)
{
    std::stringstream ss ; 
    ss << " " << msg << "[ "
       << std::setprecision(3)  
       << std::setw(fwid) << vec.x() 
       << std::setw(fwid) << vec.y() 
       << std::setw(fwid) << vec.z() 
       << "] "
       ;
    return ss.str();
}

std::string Format(const G4Track* track, const char* msg)
{
    G4int tid = track->GetTrackID();
    G4int pid = track->GetParentID();

    G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

    const G4ThreeVector& pos = track->GetPosition();
    G4double energy = track->GetKineticEnergy() ;
    G4double wavelength= h_Planck*c_light/energy ;

    std::stringstream ss ; 
    ss << "(" << msg << " ;" 
       << particleName 
       << " tid " << tid
       << " pid " << pid
       << std::setw(6) << wavelength/nm << " nm "
       << Format(pos, "pos")
       << " )"
       ;
    return ss.str();
}


std::string Format(const G4StepPoint* point, const char* msg)
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4VPhysicalVolume* pv  = point->GetPhysicalVolume();
    G4String pvName = pv ? pv->GetName() : "" ;   

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;

    const G4VProcess* process = point->GetProcessDefinedStep() ;
    const G4String& processName = process ? process->GetProcessName() : "NoProc" ; 

    G4StepStatus status = point->GetStepStatus()  ;

    std::stringstream ss ; 
    ss << "(" << msg << " ;" 
       << std::setw(15) << pvName 
       << std::setw(15) << processName 
       << std::setw(15) << OpStatusString(status)
       << Format(pos, "pos")
       << Format(dir, "dir")
       << Format(pol, "pol")
       << std::setprecision(3) << std::fixed
       << " ns " << std::setw(6) << time/ns 
       << " nm " << std::setw(6) << wavelength/nm
       << " )"
       ;

    return ss.str();
}


std::string Format(const G4Step* step, const char* msg)
{
    G4Track* track = step->GetTrack();
    G4int stepNum = track->GetCurrentStepNumber() ;
    G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

    G4StepPoint* pre  = step->GetPreStepPoint() ;
    G4StepPoint* post = step->GetPostStepPoint() ;

    std::stringstream ss ; 
    ss << "(" << msg << " ;" 
       << particleName 
       << " stepNum " << std::setw(4) << stepNum  
       << Format(track,  "tk") << "\n"
       << Format(pre,  "pre") << "\n"
       << Format(post, "post") << "\n"
       << " )"
       ;

    return ss.str();
}



