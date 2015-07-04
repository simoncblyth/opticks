#include <thrust/device_vector.h>
#include "ThrustArray.hh"
#include "NPY.hpp"
#include "Index.hpp"
#include "assert.h"

int main()
{
    typedef unsigned char S ;

    ThrustArray<S> ta(NULL, 10, 1 );   // numitems, itemsize 
    ta.dump();


    cudaDeviceSynchronize();

    return 0 ; 
}

