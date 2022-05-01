#include <cstdio>
#include "SCVD.hh"
int main()
{
    SCVD::ConfigureVisibleDevices(); 
    printf("%s\n", SCVD::Label()); 
    return 0 ; 
}
