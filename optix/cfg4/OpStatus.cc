#include "OpStatus.hh"

#include <sstream>

std::string OpStatusString(const G4StepStatus status)
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

std::string OpStatusString(const G4OpBoundaryProcessStatus status)
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


