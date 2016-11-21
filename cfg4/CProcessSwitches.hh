#pragma once

#define USE_CUSTOM_CERENKOV
#define USE_CUSTOM_SCINTILLATION
#define USE_CUSTOM_BOUNDARY
#define USE_DEBUG_TRANSPORTATION

//#define USE_POWER_THIRD_RAYLEIGH




#ifdef USE_CUSTOM_CERENKOV
class DsG4Cerenkov ; 
//class Cerenkov;
#else
class G4Cerenkov ;
#endif

#ifdef USE_CUSTOM_SCINTILLATION
class DsG4Scintillation ; 
//class Scintillation;
#else
class G4Scintillation ;
#endif

#ifdef USE_CUSTOM_BOUNDARY
class DsG4OpBoundaryProcess ; 
#else
class G4OpBoundaryProcess ; 
#endif


#ifdef USE_POWER_THIRD_RAYLEIGH
class DsG4OpRayleigh ; 
#else
class OpRayleigh ; 
#endif

#ifdef USE_DEBUG_TRANSPORTATION
class DebugG4Transportation ; 
#else
class G4Transportation ; 
#endif





