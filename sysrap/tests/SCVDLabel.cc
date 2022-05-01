#include <cstdio>
#include "SCVD.h"
int main()
{
    SCVD::ConfigureVisibleDevices(); 
    printf("%s\n", SCVD::Label()); 
    return 0 ; 
}
