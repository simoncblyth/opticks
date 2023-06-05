#include <cstdio>
#include <cassert>
#include "SCVD.h"
int main()
{
    assert(0 && "THIS APPROACH DOESNT WORK ANYMORE "); 
    SCVD::ConfigureVisibleDevices(); 
    printf("%s\n", SCVD::Label()); 
    return 0 ; 
}
