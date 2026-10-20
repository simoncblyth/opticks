
#include "OPTICKS_LOG.hh"
#include "U4GDML.h"



#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif

#include "U4Surface.h"


#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif



const char* FOLD = getenv("FOLD") ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const G4VPhysicalVolume* pv = U4GDML::Read(); 

    LOG_IF(error, pv == nullptr) << " pv null " ; 
    if(pv == nullptr) return 0 ; 





#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif

    NPFold* fold = U4Surface::MakeFold() ;  

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif



    LOG(info) << " fold " << fold ; 

    if(FOLD) 
    {
       LOG(info) << " save to FOLD " << FOLD  ;  
       fold->save(FOLD); 
    }

    return 0 ; 
}
