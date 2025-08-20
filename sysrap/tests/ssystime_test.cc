#include "ssystime.h"

int main()
{
    for(int i=0 ; i < 100 ; i++) std::cout << ssystime::local() << "\n" ;
    //for(int i=0 ; i < 100 ; i++) std::cout << ssystime::utc() << "\n" ;
    return 0 ;
}


