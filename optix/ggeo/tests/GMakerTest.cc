//  ggv --gmaker

#include "GCache.hh"
#include "GMaker.hh"
#include "GSolid.hh"
#include "GMesh.hh"

#include "glm/glm.hpp"

int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "gmaker.log", "info");

    cache->configure(argc, argv);

    GMaker* maker = new GMaker(cache);

    glm::vec4 param(0.f,0.f,0.f,100.f) ; 

    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ; 

    GSolid* solid = maker->make(0u, 'S', param, spec );

    solid->Summary();

    GMesh* mesh = solid->getMesh();

    mesh->dump();
}

