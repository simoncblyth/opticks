// name=using_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

/**

Unfortunately cannot switch beween class/struct statics (eg from TComplex to _TComplex with "using".
It only works to shorten namespace usage. 

**/

#include <cstring>
#include <cstdio>

namespace Demo
{
    static const char* Hello(){ return strdup("Demo::Hello") ; } 
}
namespace AltDemo
{
    static const char* Hello(){ return strdup("AltDemo::Hello") ; } 
}

int main()
{
    printf("Demo::Hello()    :  %s\n", Demo::Hello() ); 
    printf("AltDemo::Hello() :  %s\n", AltDemo::Hello() ); 

    {
        using Demo::Hello ; 
        printf("using Demo::Hello ; Hello() : %s\n",Hello() ); 
    }

    {
        using AltDemo::Hello ; 
        printf("using AltDemo::Hello ; Hello() : %s\n",Hello() ); 
    }

    return 0 ; 
}


