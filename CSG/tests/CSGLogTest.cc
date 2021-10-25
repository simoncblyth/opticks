#include <iostream>
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{ 
#ifdef OPTICKS_CSG
    std::cout << "OPTICKS_CSG defined " << std::endl ; 
#else
    std::cout << "OPTICKS_CSG NOT-defined " << std::endl ; 
#endif

    //OPTICKS_LOG(argc, argv); 

    OPTICKS_LOG_::Check(); 

    //LOG(info) ; 
    return 0 ; 
}
