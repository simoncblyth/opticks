#include <cstdio>
#include "qrng.h"
#include "srng.h"

int main()
{
    printf("%s\n",srng<RNG>::NAME ) ; 
    return 0 ; 
}
