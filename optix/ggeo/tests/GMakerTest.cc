//  ggv --gmaker

#include "NGLM.hpp"

#include "Opticks.hh"

#include "GMesh.hh"
#include "GSolid.hh"
#include "GMaker.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_CC.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    Opticks* opticks = new Opticks(argc, argv);

    GMaker* maker = new GMaker(opticks);

    glm::vec4 param(0.f,0.f,0.f,100.f) ; 

    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ; 

    std::vector<GSolid*> solids = maker->make(0u, 'S', param, spec );

    for(unsigned int i=0 ; i < solids.size() ; i++)
    {
        GSolid* solid = solids[i] ;

        solid->Summary();

        GMesh* mesh = solid->getMesh();

        mesh->dump();

    }

}

