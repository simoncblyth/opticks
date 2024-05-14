/**
U4TreeCreateSSimTest.cc
=========================

Default, without commandline arguments:

1. access geometry with U4VolumeMaker::PV
2. create SSim instance with empty stree and SScene
3. populate the stree and SScene instances with U4Tree::Create and SSim::initSceneFromTree
4. save the SSim to $BASE directory which appends reldir "SSim"

With any commandline argument:

1. SSim::Load from $BASE (the directory BASE should contain the "SSim" sub-dir) 
2. emit SSim::brief 


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
        sim = SSim::Load("$FOLD") ;
        assert(sim && "FOLD directory in filesystem should contain \"SSim\" subfold"); 
        LOG(info) << "SSim::Load(\"$FOLD\") " << ( sim ? sim->brief() : "-" ) ;  
    }
    else
    {
        const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
        LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;   
        if(world == nullptr) return 0 ; 

        sim = SSim::Create() ; 
        U4Tree* tr = U4Tree::Create(sim->tree, world) ; 
        assert( tr ); 

        sim->initSceneFromTree();  

        std::cerr << " save SSim to $FOLD " << std::endl ; 

        sim->save("$FOLD");  // "SSim" reldir added by the save  
        // formerly $TMP/U4TreeCreateSSimTest
    }
    LOG(info) << " sim.tree.desc " << std::endl << sim->tree->desc() ;
    return 0 ;  
}
