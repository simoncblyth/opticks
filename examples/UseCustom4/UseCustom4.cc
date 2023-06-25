
#include <iostream>
#ifdef WITH_CUSTOM4
#include "C4MultiLayrStack.h"
#endif

int main(int argc, char** argv)
{
    std::cout 
        << argv[0] 
#ifdef WITH_CUSTOM4
        << " : WITH_CUSTOM4 " 
#else
        << " : NOT-WITH_CUSTOM4 " 
#endif
        << std::endl
        ; 
    return 0 ; 
}

