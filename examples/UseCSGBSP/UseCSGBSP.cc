#include <cassert> 
#include "csgjs.hh" 

int main()
{
    csgjs_model model ;

    assert( model.indices.size() == 0 ); 
    assert( model.vertices.size() == 0 ); 

    return 0 ; 
}

