#pragma once

#define USE_CUSTOM_CERENKOV
#define USE_CUSTOM_SCINTILLATION
#define USE_CUSTOM_BOUNDARY


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



