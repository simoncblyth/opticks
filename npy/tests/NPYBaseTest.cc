// TEST=NPYBaseTest om-t 

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;


    int acv = 1001 ; 

    NPY<int>* np = NPY<int>::make(1,1,4) ; 
    np->setArrayContentVersion(acv); 

    const char* path="$TMP/npy/NPYBaseTest/acv.npy" ; 
    np->save(path); 

    NPY<int>* np2 = NPY<int>::load( path ) ; 
    int acv2 = np2->getArrayContentVersion(); 

    LOG(info) << " acv2 " << acv2 ; 

    assert( acv2 == acv ); 



    return 0 ; 
}
