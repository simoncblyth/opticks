#include <iostream>
#include <vector>

#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"

#include "NOpenMeshBoundary.hpp"
#include "NOpenMesh.hpp"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NSceneConfig.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

 


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    const char* treedir = argc > 1 ? argv[1] : "$TMP/tboolean-hybrid--/1" ;


    const char* gltfconfig = "csg_bbox_parsurf=1" ;
    const NSceneConfig* config = new NSceneConfig(gltfconfig) ; 

    NCSG* tree = NCSG::LoadTree(treedir, config );

    assert( tree );

    const nnode* root = tree->getRoot();

    NParameters* meta = tree->getMetaParameters() ;


    typedef NOpenMesh<NOpenMeshType> MESH ; 

    MESH* mesh = MESH::Make(root, meta, treedir );

    assert(mesh);



    return 0 ; 
}


