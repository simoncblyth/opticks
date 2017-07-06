#include <iostream>

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NSceneConfig.hpp"
#include "NNode.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"


void test_FromNode()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    nnode::Tests(nodes);

    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        OpticksCSG_t type = n->type ; 
        assert( type < CSG_UNDEFINED ) ;

        const char* name = n->csgname();
        assert( type > 0 && name != NULL );


        n->set_boundary(spec);

        const char* gltfconfig = "csg_bbox_parsurf=1" ;
        const NSceneConfig* config = new NSceneConfig(gltfconfig) ; 
             

        NCSG* tree = NCSG::FromNode( n , config );
        LOG(info) 
                << " node.name " << std::setw(20) << name 
                << " tree.desc " << tree->desc()
                ;

    } 
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    int verbosity = 2 ; 

    std::vector<NCSG*> trees ;

    const char* basedir = argc > 1 ? argv[0] : NULL ;

    int ntree = NCSG::DeserializeTrees( basedir, trees, verbosity );
    LOG(info) << " ntree " << ntree ; 
 

    //test_FromNode();

    return 0 ; 
}


