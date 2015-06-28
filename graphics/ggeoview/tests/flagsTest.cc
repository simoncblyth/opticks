#include "cu/photon.h"
#include <iostream>
#include <iomanip>


//#define PLACE(name, bounce)  ((unsigned long long)ffs((name))) << (((bounce) - 1)*4) 



#define DUMP(name) \
{ \
    std::cout << std::setw(18) << #name  \
              << std::dec << std::setw(5) << (name)  \
              << std::hex << std::setw(5) << (name) ; \
    unsigned long long x = ffs(name) ; \
    for(unsigned int bounce=1 ; bounce < 18 ; bounce++) \
    {  \
        unsigned int shift = (bounce-1)*4 ; \
        unsigned long long y = x << shift ; \
        std::cout << std::hex << std::setw(17) << y ; \
    } \
    std::cout << std::endl ; \
} \



int main()
{
    printf("sizeof(unsigned long long)*8 = %lu \n", sizeof(unsigned long long)*8 );

    DUMP(CERENKOV);
    DUMP(SCINTILLATION);
    DUMP(MISS);
    DUMP(BULK_ABSORB);
    DUMP(BULK_REEMIT);
    DUMP(BULK_SCATTER);
    DUMP(SURFACE_DETECT);
    DUMP(SURFACE_ABSORB);
    DUMP(SURFACE_DREFLECT);
    DUMP(SURFACE_SREFLECT);
    DUMP(BOUNDARY_REFLECT);
    DUMP(BOUNDARY_TRANSMIT);
    DUMP(NAN_ABORT);



    for(unsigned int bounce=1 ; bounce < 20 ; bounce++ )
    {
       unsigned int shift = (bounce-1)*4 ; 
       unsigned long long x = 1 << shift ;
       printf(" %2u : %2u : %20llu     %llx \n",  bounce,shift,  x, x );
    }

    unsigned long long mask = 0 ; 
    for(unsigned int bounce=1 ; bounce < 20 ; bounce++ )
    {
       unsigned int shift = (bounce-1)*4 ; 
       unsigned long long x = ffs(BOUNDARY_TRANSMIT) ; 
       unsigned long long y = x << shift ;
       mask = mask | y ;
       mask = mask | y ;
       printf(" %2u : %2u : %20llu     %20llx  %20llx  \n",  bounce,shift,  y, y, mask );
    }



    return 0 ;
}




