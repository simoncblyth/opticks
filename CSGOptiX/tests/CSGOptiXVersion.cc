#include <cstdio>
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    int vers = CSGOptiX::_OPTIX_VERSION() ; 
    printf("%d\n", vers ); 
    return 0 ; 
}


