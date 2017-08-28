#include <cassert>
#include <vector>

#include "NCSG.hpp"
#include "NNode.hpp"

#include "GParts.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "NPY_LOG.hh"


void test_FromNode()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    nnode::Tests(nodes);
    
    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    unsigned verbosity = 2 ; 

    for(unsigned i=0 ; i < nodes.size() ; i++)
    {   
        nnode* n = nodes[i] ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();
        assert( type < CSG_UNDEFINED && type > 0 && name != NULL ) ; 

        LOG(info) << "GPartsTest " 
                  << " i " << std::setw(3) << i 
                  << " type " << type
                  << " name " << name
                  ;

        n->set_boundary(spec) ; 

        const NSceneConfig* config = NULL ; 

        NCSG* tree = NCSG::FromNode( n , config  );

        GParts* pts = GParts::make( tree, spec, verbosity ) ; 
        pts->dump("GPartsTest");

    }
}


void test_save_empty()
{
    GParts pts ; 
    pts.save("$TMP/GPartsTest_test_save_empty");
}



void test_save_load()
{
    const NSceneConfig* config = NULL ; 
    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    nnode::Tests(nodes);

    nnode* n = nodes[0] ;  
    n->set_boundary(spec) ; 


    NCSG* csg = NCSG::FromNode( n, config );

    unsigned verbosity = 2 ; 
    GParts* pts = GParts::make(csg, spec, verbosity ) ; 
    pts->dump("pts");

    const char* dir = "$TMP/GPartsTest_test_save" ;
    pts->save(dir);
    GParts* pts2 = GParts::Load(dir);

    pts2->dump("pts2");


}





int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG__ ;
    NPY_LOG__ ;

    //test_save_empty();
    test_save_load();

    return 0 ;
}

