#include "OKConf.hh"
#include "OpticksCMakeConfig.hh"

unsigned OKConf::OptiXVersionInteger()
{
#ifdef OKCONF_OPTIX_VERSION_INTEGER
   return OKCONF_OPTIX_VERSION_INTEGER ;
#else 
   return 0 ; 
#endif    
}

unsigned OKConf::Geant4VersionInteger()
{
#ifdef OKCONF_GEANT4_VERSION_INTEGER
   return OKCONF_GEANT4_VERSION_INTEGER ;
#else 
   return 0 ; 
#endif    
}



