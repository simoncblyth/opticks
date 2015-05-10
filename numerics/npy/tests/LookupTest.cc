
#include "Lookup.hpp"
#include "assert.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("%s : expecting argument directory containing %s \n", argv[0], Lookup::BNAME);
        return 1 ;
    }

    Lookup lookup;
    lookup.create(argv[1]);

    lookup.dump("LookupTest");

    printf("  a => b \n");
    for(unsigned int a=0; a < 35 ; a++ )
    {
        int b = lookup.a2b(a);
        std::string aname = lookup.acode2name(a) ;
        std::string bname = lookup.bcode2name(b) ;
        printf("  %3u -> %3d  ", a, b );

        if(b < 0) printf(" %25s : WARNING failed to translate acode %u \n", aname.c_str(), a);    
        else
        {
             assert(aname == bname);
             printf(" %25s \n", aname.c_str() );
        }
    }


    return 0 ;
}
