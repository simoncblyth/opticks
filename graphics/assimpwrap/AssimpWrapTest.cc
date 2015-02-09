#include "AssimpGeometry.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"

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
    assert(query);
    assert(geokey);

    printf("argv0 %s query %s geokey %s \n", argv[0], query, geokey );

    const char* path = getenv(geokey);

    AssimpGeometry geom(path);
    geom.import();
    geom.select(query);
    geom.dumpMaterials();

    return 0 ; 
}
    

