#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "G4MultiUnion.hh"
#include "X4Solid.hh"
#include "X4SolidMaker.hh"
#include "NNode.hpp"
#include "NBBox.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv);
    ok.configure(); 

    const char* geom_default = "BoxFourBoxContiguous" ; 
    const char* geom = SSys::getenvvar("GEOM", geom_default ); 

    std::string meta ; 
    const G4VSolid* solid = X4SolidMaker::Make(geom, meta); 
    const G4MultiUnion* mun = dynamic_cast<const G4MultiUnion*>( solid ); 

    if( mun == nullptr )
    {
        LOG(fatal) << "GEOM " << geom << " is not a G4MultiUnion : ABORT " ; 
        return 0 ;
    }

    //G4cout << *mun  ; 

    const char* boundary = "" ; 
    int lvIdx = 0 ;  

    nnode* node = X4Solid::Convert(solid, &ok, boundary, lvIdx ); 

    LOG(info) << " node " << node ; 

    nbbox bb = node->bbox(); 
    bb.dump(); 


    return 0 ; 
}
