#include "Format.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include <sstream>


std::string Format(const G4ThreeVector& vec, const char* msg)
{
    std::stringstream ss ; 
    ss << " " << msg << "[ "
       << std::setprecision(3)  
       << std::setw(4) << vec.x() 
       << std::setw(4) << vec.y() 
       << std::setw(4) << vec.z() 
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

std::string Format(const G4StepStatus status)
{
    std::stringstream ss ;
    std::string s ; 
    switch(status)
    {
        case fWorldBoundary:          s="WorldBoundary"          ;break; 
        case fGeomBoundary:           s="GeomBoundary"           ;break; 
        case fAtRestDoItProc:         s="AtRestDoItProc"         ;break; 
        case fAlongStepDoItProc:      s="AlongStepDoItProc"      ;break; 
        case fPostStepDoItProc:       s="PostStepDoItProc"       ;break; 
        case fUserDefinedLimit:       s="UserDefinedLimit"       ;break; 
        case fExclusivelyForcedProc:  s="ExclusivelyForcedProc"  ;break; 
        case fUndefined:              s="Undefined"              ;break; 
        default:                      s="G4StepStatus-ERROR"     ;break;
    }
    ss << " " << s << " " ; 
    return ss.str() ;
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
       << std::setw(15) << Format(status)
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


std::string Format(const G4OpBoundaryProcessStatus status)
{
    std::stringstream ss ; 
    std::string s ; 
    switch(status)
    {
        case Undefined:s="Undefined";break;
        case Transmission:s="Transmission";break;
        case FresnelRefraction:s="FresnelRefraction";break;
        case FresnelReflection:s="FresnelReflection";break;
        case TotalInternalReflection:s="TotalInternalReflection";break;
        case LambertianReflection:s="LambertianReflection";break; 
        case LobeReflection:s="LobeReflection";break; 
        case SpikeReflection:s="SpikeReflection:";break; 
        case BackScattering:s="BackScattering";break;
        case Absorption:s="Absorption";break; 
        case Detection:s="Detection";break;
        case NotAtBoundary:s="NotAtBoundary";break;
        case SameMaterial:s="SameMaterial";break; 
        case StepTooSmall:s="StepTooSmall";break;
        case NoRINDEX:s="NoRINDEX";break;
        case PolishedLumirrorAirReflection:s="PolishedLumirrorAirReflection";break;
        case PolishedLumirrorGlueReflection:s="PolishedLumirrorGlueReflection";break;
        case PolishedAirReflection:s="PolishedAirReflection";break;
        case PolishedTeflonAirReflection:s="PolishedTeflonAirReflection";break;
        case PolishedTiOAirReflection:s="PolishedTiOAirReflection";break;
        case PolishedTyvekAirReflection:s="PolishedTyvekAirReflection";break;
        case PolishedVM2000AirReflection:s="PolishedVM2000AirReflection";break;
        case PolishedVM2000GlueReflection:s="PolishedVM2000GlueReflection";break;
        case EtchedLumirrorAirReflection:s="EtchedLumirrorAirReflection";break;
        case EtchedLumirrorGlueReflection:s="EtchedLumirrorGlueReflection";break;
        case EtchedAirReflection:s="EtchedAirReflection";break;
        case EtchedTeflonAirReflection:s="EtchedTeflonAirReflection";break;
        case EtchedTiOAirReflection:s="EtchedTiOAirReflection";break;
        case EtchedTyvekAirReflection:s="EtchedTyvekAirReflection";break;
        case EtchedVM2000AirReflection:s="EtchedVM2000AirReflection";break;
        case EtchedVM2000GlueReflection:s="EtchedVM2000GlueReflection";break;
        case GroundLumirrorAirReflection:s="GroundLumirrorAirReflection";break;
        case GroundLumirrorGlueReflection:s="GroundLumirrorGlueReflection";break;
        case GroundAirReflection:s="GroundAirReflection";break;
        case GroundTeflonAirReflection:s="GroundTeflonAirReflection";break;
        case GroundTiOAirReflection:s="GroundTiOAirReflection";break;
        case GroundTyvekAirReflection:s="GroundTyvekAirReflection";break;
        case GroundVM2000AirReflection:s="GroundVM2000AirReflection";break;
        case GroundVM2000GlueReflection:s="GroundVM2000GlueReflection";break;
        case Dichroic:s="Dichroic";break;
    }
    ss << " " << s << " " ; 
    return ss.str();
}



