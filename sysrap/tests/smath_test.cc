// name=smath_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>

#include "scuda.h"
#include "squad.h"
#include "smath.h"

int main(int argc, char** argv)
{
    typedef unsigned long long ULL ; 
    static const int N = 21 ; 
    ULL xx[N] ;
    int nn[N] ;
 
    xx[ 0] = 0x0123456789abcdefull ; nn[ 0] = 15 ; 
    xx[ 1] = 0x0023456789abcdefull ; nn[ 1] = 14 ; 
    xx[ 2] = 0x0003456789abcdefull ; nn[ 2] = 13 ; 
    xx[ 3] = 0x0000456789abcdefull ; nn[ 3] = 12 ; 
    xx[ 4] = 0x0000056789abcdefull ; nn[ 4] = 11 ; 
    xx[ 5] = 0x0000006789abcdefull ; nn[ 5] = 10 ; 
    xx[ 6] = 0x0000000789abcdefull ; nn[ 6] =  9 ; 
    xx[ 7] = 0x0000000089abcdefull ; nn[ 7] =  8 ; 
    xx[ 8] = 0x0000000009abcdefull ; nn[ 8] =  7 ; 
    xx[ 9] = 0x0000000000abcdefull ; nn[ 9] =  6 ; 
    xx[10] = 0x00000000000bcdefull ; nn[10] =  5 ; 
    xx[11] = 0x000000000000cdefull ; nn[11] =  4 ; 
    xx[12] = 0x0000000000000defull ; nn[12] =  3 ; 
    xx[13] = 0x00000000000000efull ; nn[13] =  2 ; 
    xx[14] = 0x000000000000000full ; nn[14] =  1 ; 
    xx[15] = 0x0000000000000000ull ; nn[15] =  0 ; 
    xx[16] = 0x0000d00e000a000dull ; nn[16] =  4 ; 
    xx[17] = 0x0000100000000000ull ; nn[17] =  1 ; 
    xx[18] = 0xa123456789abcdefull ; nn[18] = 16 ; 
    xx[19] = 0x1111111111111111ull ; nn[19] = 16 ; 
    xx[20] = 0x0000000000000000ull ; nn[20] =  0 ; 

    for(int i=0 ; i < N ; i++)
    {
        ULL x = xx[i] ; 
        int n = smath::count_nibbles(x) ; 
        std::cout 
            << " i " << std::setw(3)  << i  
            << " x " << std::setw(16) << std::hex << x  << std::dec
            << " n " << std::setw(3)  << n 
            << " nn[i] " << std::setw(3) << nn[i]
            << std::endl 
            ;
    }
    return 0 ; 
}
