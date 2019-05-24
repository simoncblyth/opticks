#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"

#include "NOpenMeshBoundary.hpp"
#include "NOpenMesh.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    const char* treedir = argc > 1 ? argv[1] : "$TMP/tboolean-hybrid--/1" ;

    const char* config = "csg_bbox_parsurf=1" ;

    NCSG* tree = NCSG::Load(treedir, config );

    if(!tree) 
    {
        LOG(fatal) << "NO treedir/tree " << treedir ;
        return 0 ; 
    }

    const nnode* root = tree->getRoot();

#ifdef OLD_PARAMETERS
    X_BParameters* meta = tree->getMeta(-1) ;
#else
    NMeta* meta = tree->getMeta(-1) ;
#endif

    typedef NOpenMesh<NOpenMeshType> MESH ; 

    MESH* mesh = MESH::Make(root, meta, treedir );

    assert(mesh);

    return 0 ; 
}


