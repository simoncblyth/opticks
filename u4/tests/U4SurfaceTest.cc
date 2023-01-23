
#include "OPTICKS_LOG.hh"
#include "U4GDML.h"
#include "U4Surface.h"

//#include "SNameOrder.h"

const char* FOLD = getenv("FOLD") ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const G4VPhysicalVolume* pv = U4GDML::Read(); 

    LOG_IF(error, pv == nullptr) << " pv null " ; 
    if(pv == nullptr) return 0 ; 


    /*
    const G4LogicalBorderSurfaceTable* tab = G4LogicalBorderSurface::GetSurfaceTable() ; 
    LOG(info) <<  " tab " << tab ; 

    const std::vector<G4LogicalBorderSurface*>* vec = U4Surface::PrepareBorderSurfaceVector(tab);
    LOG(info) << SNameOrder<G4LogicalBorderSurface>::Desc(*vec) ; 
    */


    NPFold* fold = U4Surface::MakeFold() ;  

    LOG(info) << " fold " << fold ; 

    if(FOLD) 
    {
       LOG(info) << " save to FOLD " << FOLD ;  
       fold->save(FOLD); 
    }

    return 0 ; 
}
