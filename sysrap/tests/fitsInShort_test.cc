// name=fitsInShort_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)

#include <cstdio>
#include <limits>

int main()
{
    int short_min = std::numeric_limits<short>::min() ; 
    int short_max = std::numeric_limits<short>::max() ; 

    printf(" short_min %d short_max %d \n", short_min, short_max ); 

    for(int i=short_min-10 ; i <= short_max+10 ; i++) 
    {
        bool ok = fitsInShort(i);         
        if( !ok || i < short_min + 100 || i > short_max - 100 )
        printf( " i %6d  0x%0.6x   -i %6d -0x %0.6x   ok %d \n", i, i,   -i, -i, ok  ); 
    }

    return 0 ; 
}
