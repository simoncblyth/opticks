#include "SSys.hh"
#include "Opticks.hh"
#include "X4Solid.hh"
#include "X4SolidMaker.hh"

#include "NNode.hpp"
#include "NBBox.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 


    const char* name = SSys::getenvvar("GEOM", "AltXJfixtureConstructionU"); 

    const G4VSolid* solid = X4SolidMaker::Make(name);  

    const char* boundary = nullptr ; 
    unsigned lvIdx = 0 ; 

    nnode* raw = X4Solid::Convert(solid, &ok, boundary, lvIdx )  ; 

    nbbox bb = raw->bbox(); 

    LOG(info) << " bb " << bb.desc() ;  



    return 0 ; 
}
