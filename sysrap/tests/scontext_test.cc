#include <cstdlib>
#include "scontext.h"

const bool VERBOSE = getenv("VERBOSE") != nullptr ; 

int main(int argc, char** argv)
{  
    char* cvd = getenv("CUDA_VISIBLE_DEVICES") ; 

    std::cout << " CUDA_VISIBLE_DEVICES : [" << ( cvd ? cvd : "-" ) << "]" << std::endl; 

    scontext sctx ; 
 
    //std::cout << ( VERBOSE ? sctx.desc() : sctx.brief() ) << std::endl ; 
    std::cout << sctx.desc() << std::endl ; 

    return 0 ; 
}
