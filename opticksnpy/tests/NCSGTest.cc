#include <iostream>

#include "NPY.hpp"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NSceneConfig.hpp"
#include "NNode.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"

void test_FromNode_Meta()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    nnode::Tests(nodes);

    const char* gltfconfig = "csg_bbox_parsurf=1" ;
    const NSceneConfig* config = new NSceneConfig(gltfconfig) ; 
    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        n->set_boundary(spec);
        NCSG* tree = NCSG::FromNode( n , config );
        LOG(info) << "test_FromNode_0 " << tree->soname() ; 
    }
}

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



void test_DeserializeTrees(int argc, char** argv )
{

    int verbosity = 2 ; 
    const char* basedir = argc > 1 ? argv[0] : NULL ;
    NCSGList* ls = NCSGList::Load( basedir, verbosity );

    if(!ls)
    {
        LOG(warning) << " no such dir " << basedir ; 
        return ; 
    }

    unsigned ntree = ls->getNumTrees();
    LOG(info) << " ntree " << ntree ; 
}


void test_setEmitconfig()
{
    const char* treedir = "$TMP/tboolean-zsphere0--/0" ; 
    LOG(info) << treedir ; 

    NCSG* csg = NCSG::LoadCSG(treedir, NULL );
    if(!csg) return ; 

    const char* ec0 = csg->getEmitConfig() ; 
    const char* ec1 = "hello:world" ; 

    csg->setEmitConfig(ec1);

    const char* ec2 = csg->getEmitConfig();

    std::cout 
        << " ec0 " << ec0
        << std::endl  
        << " ec1 " << ec1
        << std::endl  
        << " ec2 " << ec2 
        << std::endl  
        ;
       
    assert( strcmp( ec1, ec2) == 0 );
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    //test_DeserializeTrees(argc, argv);
    //test_FromNode();
    //test_FromNode_Meta();

    test_setEmitconfig();

    return 0 ; 
}


