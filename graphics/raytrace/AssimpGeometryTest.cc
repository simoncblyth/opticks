#include "AssimpGeometry.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    const char* query = getenv("RAYTRACE_QUERY");
    //if(!query) query = "__dd__Geometry__AD__lvOIL0xbf5e0b8" ;
    if(!query) query = "__dd__Geometry__PMT__lvPmtHemiCathode" ;

    printf("argv0 %s query %s \n", argv[0], query );

    const char* key = "DAE_NAME_DYB_NOEXTRA" ; 
    const char* path = getenv(key);

    AssimpGeometry geom(path);
    geom.import();
    geom.select(query);


    AssimpTree tree(geom.getRootNode());


    AssimpNode* root = tree.getRoot(); 
    printf("root %p \n", root );
    root->traverse(root);

    return 0 ; 
}
    

