// ./OKConf_test.sh 

#include <iostream>
#include <cassert>
#include "OKConf.h"

int main(int argc, char** argv)
{
    OKConf::Dump(); 
    int rc = OKConf::Check(); 

    std::cout << " OKConf::Check() " << rc << std::endl ; 

    assert( rc == 0 ); 
    return rc ;  
}

