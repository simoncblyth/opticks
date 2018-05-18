
#include <cassert>
#include <iostream>
#include "qef.h"

int main()
{
    svd::QefData q ; 

    std::cout << q.numPoints << std::endl ; 

    assert( q.numPoints == 0 );


    return 0 ; 
}
