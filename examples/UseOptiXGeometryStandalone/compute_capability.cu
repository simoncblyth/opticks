#include <stdio.h>

int main(int, char**)
{
    unsigned dev = 0 ; 
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("%d%d\n", p.major, p.minor);
}
