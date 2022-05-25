// name=stran_test ; gcc $name.cc -std=c++11 -lstdc++ -I/usr/local/cuda/include -I$OPTICKS_PREFIX/externals/glm/glm -I.. -o /tmp/$name && /tmp/$name 


#include "scuda.h"
#include "sqat4.h"
#include "stran.h"


int main()
{
    const char* a_str = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ;
    //const char* a_str = "( 0.5, 0.0, 0.0, 0.0 ) ( 0.0, 0.5, 0.0, 0.0) ( 0.0, 0.0, 0.5, 0.000) (1000.0, 1000.0,1000.0, 1.000)"  ;

    qat4* a = qat4::from_string(a_str); 

    unsigned id0[3] ;
    id0[0] = 1 ; 
    id0[1] = 10 ; 
    id0[2] = 100 ; 

    a->setIdentity( id0[0], id0[1], id0[2] );

    const qat4* i = Tran<double>::Invert( a ); 

    unsigned id1[3] ; 
    i->getIdentity( id1[0], id1[1], id1[2] ); 


    assert( id0[0] == id1[0] ); 
    assert( id0[1] == id1[1] ); 
    assert( id0[2] == id1[2] ); 


    Tran<double>* chk = Tran<double>::FromPair( a, i, 1e-3 ); 

    std::cout << chk->desc() << std::endl ; 



    return 0 ; 
}
