#include <sstream>


#include "CFG4_PUSH.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "CFG4_POP.hh"


#include "Format.hh"
#include "CStep.hh"
#include "OpStatus.hh"




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


std::string Format(const char* label, std::string pre, std::string post, unsigned int w)
{
    std::stringstream ss ; 

    ss 
       << " " 
       << std::setw(w) << label 
       << " ["
       << std::setw(w) << pre
       << "/" 
       << std::setw(w) << post
       << "]" 
       ;

    return ss.str();
}



std::string Format(const G4Track* track, const G4ThreeVector& origin, const char* msg, bool op)
{
    G4int tid = track->GetTrackID();
    G4int pid = track->GetParentID();

    G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

    const G4ThreeVector& pos = track->GetPosition();
    G4double energy = track->GetKineticEnergy() ;
    G4double wavelength= h_Planck*c_light/energy ;

    G4ThreeVector offpos = pos - origin ;  

    std::stringstream ss ; 
    ss << "(" << msg << " ;" 
       << particleName 
       << " tid " << tid
       << " pid " << pid
       << ( op ? " nm " : " keV " )
       << std::setw(6) << ( op ? wavelength/nm : energy/keV ) 
       << " mm "
       << Format(origin, "ori", 8)
       << Format(offpos, "pos", 8)
       << " )"
       ;
    return ss.str();
}

std::string Tail(const G4String& _s, unsigned int n )
{
    std::string s = _s ; 
    std::string ss = s.size() < n ? s : s.substr(s.size() - n );
    return ss ; 
}


std::string Format(const G4StepPoint* point, const G4ThreeVector& origin, const char* msg, bool op)
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4ThreeVector offpos = pos - origin ;  

    const G4String noMaterial = "noMaterial" ; 
    const G4String noProc = "noProc" ; 

    const G4Material* mat = point->GetMaterial() ;
    const G4String& matName = mat ? mat->GetName() : noMaterial ; 

    G4VPhysicalVolume* pv  = point->GetPhysicalVolume();
    G4String pvName = pv ? pv->GetName() : "" ;   

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;

    const G4VProcess* process = point->GetProcessDefinedStep() ;
    const G4String& processName = process ? process->GetProcessName() : noProc ; 

    G4StepStatus status = point->GetStepStatus()  ;

    std::stringstream ss ; 
    ss << " " << std::setw(4)  << msg  
       << " " << std::setw(25) << Tail(pvName, 25)  
       << " " << std::setw(15) << Tail(matName, 15) 
       << " " << std::setw(15) << Tail(processName, 15) 
       << std::setw(20) << OpStepString(status)
       << Format(offpos, "pos", 8)
       << Format(dir, "dir", 8)
       << Format(pol, "pol", 8)
       << std::setprecision(3) << std::fixed
       << " ns " << std::setw(6) << time/ns 
       << ( op ? " nm " : " keV " ) 
       << std::setw(6) << ( op ? wavelength/nm : energy/keV )
       ;

    return ss.str();
}


std::string Format(const G4Step* step, const G4ThreeVector& origin, const char* msg, bool op)
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
       << Format(track, origin,  "tk", op) << "\n"
       << Format(pre,  origin, "pre", op) << "\n"
       << Format(post, origin, "post", op) << "\n"
       << " )"
       ;

    return ss.str();
}


std::string Format(std::vector<const CStep*>& steps, const char* msg, bool op)
{
    unsigned int nsteps = steps.size();

    std::stringstream ss ; 

    G4ThreeVector origin ;

    for(unsigned int i=0 ; i < nsteps ; i++)
    {
         const CStep* cstep = steps[i] ;
         const G4Step* step = cstep->getStep();
         unsigned int step_id = cstep->getStepId();

        //  assert(step_id == i );  not always so ?

         G4StepPoint* pre = step->GetPreStepPoint() ;

         if(i == 0)
         {
             G4Track* track = step->GetTrack();
             G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

             const G4ThreeVector& pos = pre->GetPosition();
             origin = pos ; 

             ss << "(" << msg << " ;" 
                << particleName 
                << " nsteps " << std::setw(4) << nsteps
                << Format(track, origin, "tk", op) << "\n"
                ;
         }

         ss << Format(pre, origin, "pre", op) << "\n" ;

         if( i == nsteps - 1) 
         {
             G4StepPoint* post = step->GetPostStepPoint() ;
             ss << Format(post, origin, "post", op ) << "\n" ;
         }
    }
    return ss.str();
}






