// TEST=NSolid2Test om-t

#include "OPTICKS_LOG.hh"
#include "NSolid.hpp"
#include "NNode.hpp"


void test_is_ellipsoid()
{
    LOG(info); 

    nnode* a = NSolid::createEllipsoid( "a", 1.f, 1.f, 1.f,  -1.f, 1.f  ) ; 
    assert( a->is_ellipsoid() == false ); 
    
    nnode* b = NSolid::createEllipsoid( "b", 2.f, 1.f, 1.f,  -1.f, 1.f  ) ; 
    assert( b->is_ellipsoid() == true ); 

    nnode* c = NSolid::createEllipsoid( "c", 1.01f, 1.f, 1.f,  -1.f, 1.f  ) ; 
    assert( c->is_ellipsoid() == true ); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_is_ellipsoid();
 
    return 0 ; 
}
