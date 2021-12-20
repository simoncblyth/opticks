#include <cstdio>
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    const char* vers = CSGOptiX::_OPTIX_VERSION() ; 
    printf("%s\n", vers ); 
    return 0 ; 
}


