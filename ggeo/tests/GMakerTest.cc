//  ggv --gmaker

#include "NGLM.hpp"
#include "NSphere.hpp"

#include "Opticks.hh"

#include "GMesh.hh"
#include "GSolid.hh"
#include "GMaker.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"


void test_make(Opticks* ok)
{
    GMaker mk(ok);

    glm::vec4 param(0.f,0.f,0.f,100.f) ; 

    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ; 

    GSolid* solid = mk.make(0u, 'S', param, spec );

    solid->Summary();

    GMesh* mesh = solid->getMesh();

    mesh->dump();
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    //Opticks ok(argc, argv);

    nsphere s = make_nsphere(0,0,0,100);

    GMesh* mesh = GMaker::makeMarchingCubesMesh(s);
    mesh->dump();

}

