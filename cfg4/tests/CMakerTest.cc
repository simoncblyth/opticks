
#include <cassert>
#include "CFG4_BODY.hh"

#include "SSys.hh"
#include "NCSG.hpp"
#include "NPrimitives.hpp"

#include "BOpticksResource.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "CMaker.hh"

#include "G4VPhysicalVolume.hh"
#include "G4Sphere.hh"

#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"
#include "PLOG.hh"



void test_load_csg(CMaker& mk, int argc, char** argv)
{
    BOpticksResource okr ;  
    std::string treedir = okr.getDebuggingTreedir(argc, argv);

    const char* config = NULL ; 
    NCSG* csg = NCSG::LoadCSG( treedir.c_str(), config );  
    if(!csg) return ; 

    csg->dump();

    G4VSolid* solid = mk.makeSolid(csg);
    assert(solid); 
}

void test_make_csg(CMaker& mk)
{
    nsphere sp = make_sphere();
    sp.set_boundary("Dummy"); 

    NCSG* csg = NCSG::FromNode(&sp, NULL);
    if(!csg) return ; 
 
    csg->dump();

    G4VSolid* solid = mk.makeSolid(csg);
    assert(solid); 

    G4Sphere* sp_ = dynamic_cast<G4Sphere*>(solid);
    assert(sp_);

    double radius = sp_->GetOuterRadius();
    assert( radius == 100. );

    LOG(info) << " sp " << *sp_ ; 

}


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

    //test_load_csg(mk, argc, argv);
    test_make_csg(mk);





    return 0 ; 
} 
