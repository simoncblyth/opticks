
#include <cassert>
#include "CFG4_BODY.hh"

#include "SSys.hh"
#include "NCSG.hpp"

#include "BOpticksResource.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "CMaker.hh"

#include "G4VPhysicalVolume.hh"

#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv)

    BRAP_LOG__ ; 
    NPY_LOG__ ; 
    GGEO_LOG__ ; 
    CFG4_LOG__ ; 

    LOG(info) << argv[0] ; 


    unsigned verbosity = SSys::getenvint("VERBOSITY", 1);

    Opticks ok(argc, argv);
    CMaker mk(&ok, verbosity );

    // TODO: get from Opticks rather than the debugging way via IDFOLD
    BOpticksResource okr ;  
    std::string treedir = okr.getDebuggingTreedir(argc, argv);

    const char* config = NULL ; 
    NCSG* csg = NCSG::LoadCSG( treedir.c_str(), config );  
    if(!csg) return 0 ; 

    csg->dump();

    G4VSolid* solid = mk.makeSolid(csg);
    assert(solid); 


    return 0 ; 
} 
