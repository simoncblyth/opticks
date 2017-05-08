
#include <cassert>
#include <vector>

#include "GParts.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

#include "NCSG.hpp"
#include "NNode.hpp"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    nnode::Tests(nodes);
    
    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {   
        nnode* n = *it ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();
        assert( type < CSG_UNDEFINED && type > 0 && name != NULL ) ; 

        NCSG* tree = NCSG::FromNode( n , spec );  // TODO: eliminate spec from NCSG 

        GParts* pts = GParts::make( tree, spec ) ; 
        pts->dump("GPartsTest");

    }

    return 0 ;
}

