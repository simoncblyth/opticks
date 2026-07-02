/**
OKConf_CUDART_test.cc
=======================

~/o/okconf/tests/OKConf_CUDART_test.sh

**/

#include <iostream>
#include <cassert>
#include "OKConf_CUDART.h"

int main(int argc, char** argv)
{
    std::cout << OKConf_CUDART::Desc();
    int rc = OKConf_CUDART::Check();

    std::cout << " OKConf_CUDART::Check() " << rc << std::endl ;

    assert( rc == 0 );
    return rc ;
}

