/**
U4TreeCreateSSimTest.cc
=========================

**/

#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "U4VolumeMaker.hh"
#include "U4Tree.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim* sim = nullptr ; 

    if( argc > 1 )
    {
        sim = SSim::Load("$BASE") ;
        assert(sim && "BASE directory should contain \"SSim\" subfold"); 
        LOG(info) << "SSim::Load(\"$BASE\") " << ( sim ? sim->brief() : "-" ) ;  
    }
    else
    {
        const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
        LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;   
        if(world == nullptr) return 0 ; 

        sim = SSim::Create() ; 
        U4Tree::Create(sim->tree, world) ; 

        std::cerr << " save SSim to $TMP/U4TreeCreateSSimTest " << std::endl ; 

        sim->save("$TMP/U4TreeCreateSSimTest"); 
    }
    LOG(info) << " sim.tree.desc " << std::endl << sim->tree->desc() ;
    return 0 ;  
}
