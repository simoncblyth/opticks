#include <cstdlib>
#include "scontext.h"

const bool VERBOSE = getenv("VERBOSE") != nullptr ; 

int main(int argc, char** argv)
{  
    scontext sctx ; 
 
    std::cout << ( VERBOSE ? sctx.desc() : sctx.brief() ) << std::endl ; 

    return 0 ; 
}
