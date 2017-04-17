#include <iostream>

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"



void test_Deserialize(const char* base)
{
    int verbosity = 1 ; 
    std::vector<NCSG*> trees ;
    NCSG::Deserialize( base, trees, verbosity );
    LOG(info) << "test_Deserialize " << base << " found trees : " << trees.size() ;
    for(unsigned i=0 ; i < trees.size() ; i++) trees[i]->dump("NCSGTest dump");
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


        NCSG* tree = NCSG::FromNode( n , spec );
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

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    test_Deserialize( argc > 1 ? argv[1] : "$TMP/csg_py") ; 
    //test_FromNode();

    return 0 ; 
}


