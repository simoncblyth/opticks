#include <cassert>
#include <vector>

#include "NCSG.hpp"
#include "NNode.hpp"

#include "GParts.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG__ ;

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

        NCSG* tree = NCSG::FromNode( n , spec );  // TODO: eliminate spec from NCSG 

        GParts* pts = GParts::make( tree, spec, verbosity ) ; 
        pts->dump("GPartsTest");


    }

    return 0 ;
}

