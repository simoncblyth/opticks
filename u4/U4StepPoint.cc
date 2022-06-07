#include <sstream>

#include "G4StepPoint.hh"
#include "G4VProcess.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4OpBoundaryProcess.hh"

#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"

#include "U4StepStatus.h"
#include "U4OpBoundaryProcess.hh"
#include "U4OpBoundaryProcessStatus.h"
#include "U4StepPoint.hh"

/**
U4StepPoint::Update
---------------------

* cf CWriter::writeStepPoint_

**/

void U4StepPoint::Update(sphoton& photon, const G4StepPoint* point)  // static
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& mom = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    
    photon.pos.x = pos.x(); 
    photon.pos.y = pos.y(); 
    photon.pos.z = pos.z(); 
    photon.time  = time/ns ; 

    photon.mom.x = mom.x(); 
    photon.mom.y = mom.y(); 
    photon.mom.z = mom.z(); 
    //photon.iindex = 0u ; 

    photon.pol.x = pol.x(); 
    photon.pol.y = pol.y(); 
    photon.pol.z = pol.z(); 
    photon.wavelength = wavelength/nm ; 
}

unsigned U4StepPoint::ProcessDefinedStepType(const G4StepPoint* point) // static
{
    const G4VProcess* process = point->GetProcessDefinedStep() ;
    if(process == nullptr) return U4StepPoint_NoProc ; 
    const G4String& processName = process->GetProcessName() ; 
    return ProcessDefinedStepType( processName.c_str() ); 
}

unsigned U4StepPoint::ProcessDefinedStepType(const char* name) // static 
{
    unsigned type = U4StepPoint_Undefined ;
    if(strcmp(name, NoProc_) == 0 )         type = U4StepPoint_NoProc ; 
    if(strcmp(name, Transportation_) == 0 ) type = U4StepPoint_Transportation ;
    if(strcmp(name, OpRayleigh_) == 0)      type = U4StepPoint_OpRayleigh ;   
    if(strcmp(name, OpAbsorption_) == 0)    type = U4StepPoint_OpAbsorption ;   
    if(strcmp(name, OTHER_) == 0)           type = U4StepPoint_OTHER ;   
    return type ; 
}

const char* U4StepPoint::ProcessDefinedStepTypeName( unsigned type ) // static
{
    const char* s = nullptr ; 
    switch(type)
    {
        case U4StepPoint_Undefined:      s = Undefined_      ; break ;  
        case U4StepPoint_NoProc:         s = NoProc_         ; break ;
        case U4StepPoint_Transportation: s = Transportation_ ; break ;
        case U4StepPoint_OpRayleigh:     s = OpRayleigh_     ; break ;
        case U4StepPoint_OpAbsorption:   s = OpAbsorption_   ; break ;  
        default:                         s = OTHER_          ; break ; 
    }
    return s ; 
}

/**
U4StepPoint::Flag
------------------

Adapted from cfg4/OpStatus.cc:OpStatus::OpPointFlag

**/
unsigned U4StepPoint::Flag(const G4StepPoint* point)
{
    G4StepStatus status = point->GetStepStatus()  ;
    unsigned proc = ProcessDefinedStepType(point); 
    unsigned flag = 0 ; 

    if( status == fPostStepDoItProc && proc == U4StepPoint_OpAbsorption )
    {
        flag = BULK_ABSORB ; 
    }
    else if( status == fPostStepDoItProc && proc == U4StepPoint_OpRayleigh )
    {
        flag = BULK_SCATTER ; 
    }
    else if( status == fGeomBoundary && proc == U4StepPoint_Transportation )
    {
        unsigned bstat = U4OpBoundaryProcess::GetStatus(); 
        flag = BoundaryFlag(bstat) ;   
    }
    else if( status == fWorldBoundary && proc == U4StepPoint_Transportation )
    {
        flag = MISS ; 
    }
    return flag ; 
}

unsigned U4StepPoint::BoundaryFlag(unsigned status)
{
    unsigned flag = 0 ; 
    switch(status)
    {   
        case FresnelRefraction:
        case SameMaterial:
                               flag=BOUNDARY_TRANSMIT;
                               break;
        case TotalInternalReflection:
        case       FresnelReflection:
                               flag=BOUNDARY_REFLECT;
                               break;
        case StepTooSmall:
                               flag=NAN_ABORT;
                               break;
        case Absorption:
                               flag=SURFACE_ABSORB ; 
                               break;
        case Detection:
                               flag=SURFACE_DETECT ; 
                               break;
        case SpikeReflection:
                               flag=SURFACE_SREFLECT ; 
                               break;
        case LobeReflection:
        case LambertianReflection:
                               flag=SURFACE_DREFLECT ; 
                               break;
        case NoRINDEX:
                               flag=NAN_ABORT;
                               break;
        case Undefined:
        case BackScattering:
        case NotAtBoundary:
        case Transmission:
        case PolishedLumirrorAirReflection:
        case PolishedLumirrorGlueReflection:
        case PolishedAirReflection:
        case PolishedTeflonAirReflection:
        case PolishedTiOAirReflection:
        case PolishedTyvekAirReflection:
        case PolishedVM2000AirReflection:
        case PolishedVM2000GlueReflection:
        case EtchedLumirrorAirReflection:
        case EtchedLumirrorGlueReflection:
        case EtchedAirReflection:
        case EtchedTeflonAirReflection:
        case EtchedTiOAirReflection:
        case EtchedTyvekAirReflection:
        case EtchedVM2000AirReflection:
        case EtchedVM2000GlueReflection:
        case GroundLumirrorAirReflection:
        case GroundLumirrorGlueReflection:
        case GroundAirReflection:
        case GroundTeflonAirReflection:
        case GroundTiOAirReflection:
        case GroundTyvekAirReflection:
        case GroundVM2000AirReflection:
        case GroundVM2000GlueReflection:
        case Dichroic:
                               flag=0;   // leads to bad flag asserts
                               break;
    }
    return flag ;
}


std::string U4StepPoint::Desc(const G4StepPoint* point)
{
    G4StepStatus status = point->GetStepStatus()  ;
    const char* statusName = U4StepStatus::Name(status); 

    unsigned proc = ProcessDefinedStepType(point); 
    const char* procName = ProcessDefinedStepTypeName(proc); 

    unsigned bstat = U4OpBoundaryProcess::GetStatus(); 
    const char* bstatName = U4OpBoundaryProcessStatus::Name(bstat); 

    unsigned flag = Flag(point); 
    const char* flagName = OpticksPhoton::Flag(flag); 

    std::stringstream ss ; 
    ss << "U4StepPoint::Desc" 
       << " proc " << proc
       << " procName " << procName 
       << " status " << status
       << " statusName " << statusName 
       << " bstat " << bstat
       << " bstatName " << bstatName
       << " flag " << flag 
       << " flagName " << flagName
       ;
    std::string s = ss.str(); 
    return s ; 
}

