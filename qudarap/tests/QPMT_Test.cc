/**
QPMT_Test.cc : standalone built variant of om built QPMTTest.cc
=================================================================
**/

#include "QPMT.hh"

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "get_jpmt_fold.h"
#include "QPMTTest.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    std::cout << " Before: " << QPMT<float>::Desc() << std::endl ; 

    const NPFold* jpmt = get_jpmt_fold(); 

    QPMTTest<float> t(jpmt); 
    NPFold* f = t.serialize(); 
    cudaDeviceSynchronize();
    f->save("$FOLD"); 

    std::cout << " Final: " << QPMT<float>::Desc() << std::endl ; 

    return 0 ; 
}
