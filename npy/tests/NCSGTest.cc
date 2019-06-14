#include <iostream>

#include "NPY.hpp"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NSceneConfig.hpp"
#include "NNode.hpp"
#include "NNodeSample.hpp"

#include "OPTICKS_LOG.hh"

void test_Adopt_Meta()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    const char* gltfconfig = "csg_bbox_parsurf=1" ;
    const NSceneConfig* config = new NSceneConfig(gltfconfig) ; 
    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        n->set_boundary(spec);

        unsigned soIdx = 0 ; 
        unsigned lvIdx = 0 ; 

        NCSG* tree = NCSG::Adopt( n , config , soIdx, lvIdx );
        LOG(info) << "test_Adopt_0 " << tree->get_soname() ; 
    }
}

void test_Adopt()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

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

        unsigned soIdx = 0 ; 
        unsigned lvIdx = 0 ; 
             
        NCSG* tree = NCSG::Adopt( n , config, soIdx, lvIdx );
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

    NCSG* csg = NCSG::Load(treedir);
    if(!csg) return ; 

    const char* ec0 = csg->get_emitconfig() ; 
    const char* ec1 = "hello:world" ; 

    csg->set_emitconfig(ec1);

    const char* ec2 = csg->get_emitconfig();

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
    OPTICKS_LOG(argc, argv);

    //test_DeserializeTrees(argc, argv);
    //test_Adopt();
    //test_Adopt_Meta();
    test_setEmitconfig();

    return 0 ; 
}


