#include <iostream>
#include <cassert>
#include "OKConf.hh"

int main()
{
    OKConf::Dump(); 
    int rc = OKConf::Check(); 

    std::cout << " OKConf::Check() " << rc << std::endl ; 

    assert( rc == 0 ); 
    return rc ;  
}


