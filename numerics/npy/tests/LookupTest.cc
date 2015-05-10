
#include "Lookup.hpp"


int main()
{
    Lookup lookup;

    const char* apath = "/tmp/ChromaMaterialMap.json" ;
    const char* aprefix = "/dd/Materials/" ;

    lookup.create(apath, aprefix, apath, aprefix);

    for(unsigned int a=0; a < 35 ; a++ )
    {
        int b = lookup.a2b(a);
        printf("  a => b %u -> %d \n", a, b );
    }


    return 0 ;
}
