// name=stime_test ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include "stime.h"

void test_Time()
{
    stime::Time stamp ; 
    stime::init(&stamp);  

    std::cout << stime::Desc(&stamp) << std::endl ;  
}

void test_Format()
{
    std::cout << "stime::Stamp "  << stime::Stamp() << std::endl ; 
    std::cout << "stime::Now    " << stime::Now() << std::endl ; 
    std::cout << "stime::Format " << stime::Format() << std::endl ; 

}


int main(int argc, char** argv)
{

    test_Format(); 
    return 0 ; 
}
