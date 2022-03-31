// name=sflowTest ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>

#include "sflow.h"

int main()
{
    for(unsigned i=0 ; i <= LAST ; i++ ) std::cout << std::setw(3) << i << " " << sflow::desc(i) << std::endl ; 
    return 0 ; 
} 


