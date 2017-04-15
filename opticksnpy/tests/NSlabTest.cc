#include "NSlab.hpp"

#include <iostream>
#include <iomanip>

int main()
{
    nslab s = make_nslab( 0,0,1, 10) ; 

    for(int i=-20 ; i < 20 ; i++)
        std::cout << std::setw(4) << i << " " << s(0.f,0.f,i) << std::endl ;  


    return 0 ; 
}
