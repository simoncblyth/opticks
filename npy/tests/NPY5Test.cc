// om-;TEST=NPY5Test om-t

#include "OPTICKS_LOG.hh"
#include "SStr.hh"
#include "SSys.hh"
#include "NPY.hpp"

/**
test_row_major_serialization
-----------------------------

The slowest dimension in the serialization is the first.

* https://en.wikipedia.org/wiki/Row-_and_column-major_order

* Row-major order is used in C/C++/Objective-C (for C-style arrays)
* Row-major order is the default in NumPy

**/

void test_row_major_serialization()
{
    int ni = 8 ; 
    int nj = 16 ; 

    NPY<int>* a = NPY<int>::make(ni,nj) ; 
    a->fillIndexFlat(); 
    a->dump(); 

    const char* path = "$TMP/npy/tests/NPY5Test/a.npy" ;
    a->save(path);
    LOG(info) << path ; 

    int count = 0 ; 
    for(int i=0 ; i < ni ; i++){
        for(int j=0 ; j < nj ; j++)
        {
            int value = a->getValue(i,j,0); 
            std::cout << std::setw(3) << value << " " ; 
            assert( value == count ); 
            count += 1 ;     
        }
        std::cout << std::endl ; 
    }
    SSys::run(SStr::Concat("xxd ",path));
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_row_major_serialization(); 

    return 0 ; 
}


// om-;TEST=NPY5Test om-t
