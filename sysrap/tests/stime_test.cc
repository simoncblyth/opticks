// name=stime_test ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include "stime.h"

int main(int argc, char** argv)
{
    stime::Time stamp ; 
    stime::init(&stamp);  

    std::cout << stime::Desc(&stamp) << std::endl ;  

    return 0 ; 
}
