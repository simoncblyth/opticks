// om-;TEST=NPY5Test om-t

#include "OPTICKS_LOG.hh"
#include "SStr.hh"
#include "SSys.hh"
#include "SProc.hh"
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



void test_make(int n)
{
    LOG(info); 
    float vm0 = SProc::VirtualMemoryUsageMB() ; 
    for(int i=0 ; i < n ; i++)
    {
        NPY<float>* a = NPY<float>::make(10000, 4 ); 
        a->zero(); 
        a->reset();    // reset alone not enough, need to delete too 

        float vm = SProc::VirtualMemoryUsageMB() ; 
        float dv = vm - vm0 ; 
        std::cout 
            << std::setw(6) << i 
            << " : "
            << std::setw(6) << vm
            << " : "
            << std::setw(6) << dv
            << " : "
            << std::setw(6) << a->capacity()
            << std::endl
            ; 
        
        //delete a ;  
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    int n = argc > 1 ? atoi(argv[1]) : 10 ; 

    //test_row_major_serialization(); 

    test_make(n); 

    return 0 ; 
}


// om-;TEST=NPY5Test om-t
