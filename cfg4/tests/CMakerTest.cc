
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

#include "OPTICKS_LOG.hh"



void test_load_csg(CMaker& mk, int argc, char** argv)
{
    BOpticksResource okr ;  
    std::string treedir = okr.getDebuggingTreedir(argc, argv);

    NCSG* csg = NCSG::Load( treedir.c_str() );  
    if(!csg) return ; 

    csg->dump();

    G4VSolid* solid = mk.makeSolid(csg);
    assert(solid); 
}

void test_make_csg(CMaker& mk)
{
    nsphere sp = make_sphere();
    sp.set_boundary("Dummy"); 

    NCSG* csg = NCSG::Adopt(&sp);
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
    OPTICKS_LOG(argc, argv)

    LOG(info) << argv[0] ; 

    unsigned verbosity = SSys::getenvint("VERBOSITY", 1);

    Opticks ok(argc, argv);
    CMaker mk(&ok, verbosity );

    //test_load_csg(mk, argc, argv);
    test_make_csg(mk);


    return 0 ; 
} 
