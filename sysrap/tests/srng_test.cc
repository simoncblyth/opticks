/**

~/o/sysrap/tests/srng_test.sh 

**/

#include <iostream>
#include "srng.h"

using RNG0 = curandStateXORWOW ; 
using RNG1 = curandStatePhilox4_32_10 ; 

#ifdef WITH_CURANDLITE
using RNG2 = curandStatePhilox4_32_10_OpticksLite ; 
#endif

int main()
{
    std::cout << srng_Desc<RNG0>() << "\n\n" ; 
    std::cout << srng_Desc<RNG1>() << "\n\n" ; 
#ifdef WITH_CURANDLITE
    std::cout << srng_Desc<RNG2>() << "\n\n" ; 
#endif

    return 0 ;  
}
