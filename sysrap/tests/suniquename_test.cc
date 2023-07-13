// name=suniquename_test  ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>
#include "suniquename.h"

int main()
{
    std::vector<std::string> names ; 
    int idx ; 

    idx = suniquename::Add("red", names ); assert( idx == 0 ); 
    idx = suniquename::Add("red", names ); assert( idx == 0 ); 
    idx = suniquename::Add("red", names ); assert( idx == 0 ); 

    idx = suniquename::Add("green", names ); assert( idx == 1 ); 
    idx = suniquename::Add("green", names ); assert( idx == 1 ); 
    idx = suniquename::Add("green", names ); assert( idx == 1 ); 

    idx = suniquename::Add("blue", names ); assert( idx == 2 ); 
    idx = suniquename::Add("blue", names ); assert( idx == 2 ); 
    idx = suniquename::Add("blue", names ); assert( idx == 2 ); 

    idx = suniquename::Add("red", names ); assert( idx == 0 ); 
    idx = suniquename::Add("red", names ); assert( idx == 0 ); 
    idx = suniquename::Add("red", names ); assert( idx == 0 ); 

    idx = suniquename::Add("cyan", names ); assert( idx == 3 ); 
    idx = suniquename::Add("cyan", names ); assert( idx == 3 ); 
    idx = suniquename::Add("cyan", names ); assert( idx == 3 ); 

    idx = suniquename::Add("red", names ); assert( idx == 0 ); 
    idx = suniquename::Add("green", names ); assert( idx == 1 ); 
    idx = suniquename::Add("blue", names ); assert( idx == 2 ); 
    idx = suniquename::Add("cyan", names ); assert( idx == 3 ); 

    std::cout << suniquename::Desc(names) ; 

    return 0 ; 
}
