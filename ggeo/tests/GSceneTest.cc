
#include <set>
#include <string>

#include "NScene.hpp"

#include "Opticks.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include "GScene.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GGEO_LOG__ ;

    Opticks ok(argc, argv);

    GGeo gg(&ok);
    gg.loadFromCache();
    gg.dumpStats();

    const char* base = "$TMP/nd" ;
    const char* name = "scene.gltf" ; 

    NScene ns(base, name); 
    GScene gs(&gg, &ns) ; 


    return 0 ; 
}


