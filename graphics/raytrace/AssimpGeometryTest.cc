#include "AssimpGeometry.hh"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    const char* query = getenv("RAYTRACE_QUERY");
    if(!query) query = "__dd__Geometry__AD__lvOIL0xbf5e0b8" ;
    printf("argv0 %s query %s \n", argv[0], query );

    const char* key = "DAE_NAME_DYB_NOEXTRA" ; 
    const char* path = getenv(key);

    AssimpGeometry geom(path, query );
    geom.import();

    aiNode* node = geom.searchNode(query);
    printf("query %s => node %p \n", query, node );

    return 0 ; 
}
    

