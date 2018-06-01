#include <iostream>
#include <cassert>

#include "SAr.hh"
#include "SSys.hh"

int main(int argc, char** argv)
{
    std::cout << "start" << std::endl ; 

    SAr a(argc, argv );
    std::cout << "a instanciated " << std::endl ; 
    a.dump();
    //a.dump();
    std::cout << "a dumped " << std::endl ; 


    const char* key = "SAR_TEST" ; 
    const char* val = "--trace --SYSRAP warning red green blue" ; 
    bool overwrite = true ; 
    SSys::setenvvar( key, val , overwrite ) ;  

    SAr b(0,0, key, ' ') ;
    std::cout << "b instanciated " << std::endl ; 
    b.dump() ; 
    std::cout << "b dumped " << std::endl ; 

    assert( b._argc == 7 ); 


    return 0 ; 
}
