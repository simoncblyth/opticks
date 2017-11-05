#include "CFG4_BODY.hh"
#include <sstream>

#include "G4StepPoint.hh"
#include "Opticks.hh"

#include "OpStatus.hh"
#include "CBoundaryProcess.hh"

#include "PLOG.hh"


std::string OpStepString(const G4StepStatus status)
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
    ss << s ; 
    return ss.str() ;
}


#ifdef USE_CUSTOM_BOUNDARY
std::string OpBoundaryAbbrevString(const DsG4OpBoundaryProcessStatus status)
#else
std::string OpBoundaryAbbrevString(const G4OpBoundaryProcessStatus status)
#endif
{
    std::stringstream ss ; 
    std::string s ; 
    switch(status)
    {
        case Undefined:s="Und";break;
        case FresnelRefraction:s="FrT";break;
        case FresnelReflection:s="FrR";break;
        case TotalInternalReflection:s="TIR";break;
        case LambertianReflection:s="LaR";break; 
        case LobeReflection:s="LoR";break; 
        case SpikeReflection:s="SpR";break; 
        case BackScattering:s="BkS";break;
        case Absorption:s="Abs";break; 
        case Detection:s="Det";break;
        case NotAtBoundary:s="NAB";break;
        case SameMaterial:s="SAM";break; 
        case StepTooSmall:s="STS";break;
        case NoRINDEX:s="NRI";break;
#ifdef USE_CUSTOM_BOUNDARY
#else
        case Transmission:s="Tra";break;
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
#endif
    }
    ss << s ; 
    return ss.str();
}



#ifdef USE_CUSTOM_BOUNDARY
std::string OpBoundaryString(const DsG4OpBoundaryProcessStatus status)
#else
std::string OpBoundaryString(const G4OpBoundaryProcessStatus status)
#endif
{
    std::stringstream ss ; 
    std::string s ; 
    switch(status)
    {
        case Undefined:s="Undefined";break;
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
#ifdef USE_CUSTOM_BOUNDARY
#else
        case Transmission:s="Transmission";break;
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
#endif
    }
    ss << s ; 
    return ss.str();
}





#ifdef USE_CUSTOM_BOUNDARY
unsigned int OpBoundaryFlag(const DsG4OpBoundaryProcessStatus status)
#else
unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status)
#endif
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
        case Undefined:
        case BackScattering:
        case NotAtBoundary:
        case NoRINDEX:

#ifdef USE_CUSTOM_BOUNDARY
#else
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
#endif
                      flag=0;
                      break;
    }
    return flag ; 
}


#ifdef USE_CUSTOM_BOUNDARY
unsigned int OpPointFlag(const G4StepPoint* point, const DsG4OpBoundaryProcessStatus bst, CStage::CStage_t stage)
#else
unsigned int OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst, CStage::CStage_t stage)
#endif
{
    G4StepStatus status = point->GetStepStatus()  ;
    // TODO: cache the relevant process objects, so can just compare pointers ?
    const G4VProcess* process = point->GetProcessDefinedStep() ;
    const G4String& processName = process ? process->GetProcessName() : "NoProc" ; 

    bool transportation = strcmp(processName,"Transportation") == 0 ;
    bool scatter = strcmp(processName, "OpRayleigh") == 0 ; 
    bool absorption = strcmp(processName, "OpAbsorption") == 0 ;

    unsigned flag(0);

    // hmm stage and REJOINing look kinda odd here, do elsewhere ?
    // moving it first, breaks seqhis matching for multi-RE lines 

    if(absorption && status == fPostStepDoItProc )
    {
        flag = BULK_ABSORB ;
    }
    else if(scatter && status == fPostStepDoItProc )
    {
        flag = BULK_SCATTER ;
    }
    else if( stage == CStage::REJOIN )  
    { 
        flag = BULK_REEMIT ;  
    } 
    else if(transportation && status == fGeomBoundary )
    {
        flag = OpBoundaryFlag(bst) ; // BOUNDARY_TRANSMIT/BOUNDARY_REFLECT/NAN_ABORT/SURFACE_ABSORB/SURFACE_DETECT/SURFACE_DREFLECT/SURFACE_SREFLECT
    } 
    else if(transportation && status == fWorldBoundary )
    {
        //flag = SURFACE_ABSORB ;   // former kludge for fWorldBoundary - no surface handling yet 
        flag = MISS ; 
    }
    else
    {
        LOG(warning) << " OpPointFlag ZERO  " 
                     << " proceesDefinedStep? " << processName 
                     << " stage " << CStage::Label(stage)
                     << " status " << OpStepString(status)
                     ;
        //assert(0);
    }
    return flag ; 
}



