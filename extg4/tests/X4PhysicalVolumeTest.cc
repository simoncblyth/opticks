#include "GMaterialLib.hh"

#include "X4PhysicalVolume.hh"
#include "X4MaterialTable.hh"
#include "OpNoviceDetectorConstruction.hh"
#include "Opticks.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);
    Opticks ok(argc, argv);
    ok.configure();

    LOG(info) << " ok.verbosity " << ok.getVerbosity() ; 

    OpNoviceDetectorConstruction ondc ; 

    G4VPhysicalVolume* top = ondc.Construct() ;     
    assert(top);  

    GGeo* ggeo = X4PhysicalVolume::Convert(top) ;   
    assert(ggeo);  

    //GMaterialLib* mlib = X4MaterialTable::Convert() ; 
    //assert( mlib->getNumMaterials() == 2 ); 
    //mlib->dump();

    return 0 ; 
}
