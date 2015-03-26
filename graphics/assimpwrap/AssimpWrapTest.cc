#include "AssimpGeometry.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpGGeo.hh"

#include "GGeo.hh"
#include "GSubstanceLib.hh"
#include "GMergedMesh.hh"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
   comparing with pycollada

      g4daenode.sh -i --daepath dyb_noextra


*/

int main(int argc, char* argv[])
{
    const char* query = getenv("ASSIMPWRAP_QUERY");
    const char* geokey = getenv("ASSIMPWRAP_GEOKEY");
    const char* material = getenv("ASSIMPWRAP_MATERIAL");
    const char* ggctrl = getenv("ASSIMPWRAP_GGCTRL");

    assert(query);
    assert(geokey);
    assert(material);

    printf("argv0 %s query %s geokey %s material %s \n", argv[0], query, geokey, material );
    const char* path = getenv(geokey);

    AssimpGeometry ageo(path);
    ageo.import();
    AssimpSelection* selection = ageo.select(query);

    AssimpGGeo agg(ageo.getTree(), selection); 
    GGeo* ggeo = agg.convert(ggctrl);

    //ggeo->Summary("main");    

    GSubstanceLib* lib = ggeo->getSubstanceLib();
    lib->Summary("GSubstanceLib");

    GMergedMesh* mm = ggeo->getMergedMesh();
    mm->Summary("GMergedMesh");
  

    return 0 ; 
}
    

