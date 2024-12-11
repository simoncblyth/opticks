/**

~/o/qudarap/tests/QRngSpec_test.sh 

**/

#include <iostream>
#include "QRngSpec.h"

using RNG0 = curandStateXORWOW ; 
using RNG1 = curandStatePhilox4_32_10 ; 

#ifdef WITH_CURANDLITE
using RNG2 = curandStatePhilox4_32_10_OpticksLite ; 
#endif

int main()
{
    printf(" QRngSpec<RNG0>::NAME %s\n",   QRngSpec<RNG0>::NAME ); 
    printf(" QRngSpec<RNG0>::CODE %c\n",   QRngSpec<RNG0>::CODE ); 
    printf(" QRngSpec<RNG0>::SIZE %d\n\n", QRngSpec<RNG0>::SIZE ); 


    printf(" QRngSpec<RNG1>::NAME %s\n",   QRngSpec<RNG1>::NAME ); 
    printf(" QRngSpec<RNG1>::CODE %c\n",   QRngSpec<RNG1>::CODE ); 
    printf(" QRngSpec<RNG1>::SIZE %d\n\n", QRngSpec<RNG1>::SIZE ); 

#ifdef WITH_CURANDLITE
    printf(" QRngSpec<RNG2>::NAME %s\n",   QRngSpec<RNG2>::NAME ); 
    printf(" QRngSpec<RNG2>::CODE %c\n",   QRngSpec<RNG2>::CODE ); 
    printf(" QRngSpec<RNG2>::SIZE %d\n\n", QRngSpec<RNG2>::SIZE ); 
#endif

/*
    std::cout << QRngSpec<RNG0>::Desc() << "\n" ; 
    std::cout << QRngSpec<RNG1>::Desc() << "\n" ; 
#ifdef WITH_CURANDLITE
    std::cout << QRngSpec<RNG2>::Desc() << "\n" ; 
#endif
*/

    return 0 ;  
}
