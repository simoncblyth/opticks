#include "GMaterialLib.hh"

#include "X4PhysicalVolume.hh"
#include "X4MaterialTable.hh"
#include "OpNoviceDetectorConstruction.hh"
#include "Opticks.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    // To setup resources appropriately within Opticks
    // when using the live G4 to GGeo approach  
    // need to defer Opticks instanciation
    // until after OpticksId::SetId.  This ordering/setup
    // is done in the X4PhysicalVolume ctor.
    //

    /*
    Opticks ok(argc, argv);
    ok.configure();
    LOG(info) << " ok.verbosity " << ok.getVerbosity() ; 
    */
   

    OpNoviceDetectorConstruction ondc ; 

    G4VPhysicalVolume* top = ondc.Construct() ;     
    assert(top);  

    // NB as this test does not bootup a full Geant4 environment
    //    cannot grab top via the navigator singleton  

    const char* id = X4PhysicalVolume::Id(top) ;      
    LOG(info) << " id " << id ;  

    GGeo* ggeo = X4PhysicalVolume::Convert(top) ;   
    assert(ggeo);  


    return 0 ; 
}
