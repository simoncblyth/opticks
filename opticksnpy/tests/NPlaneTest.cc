#include "NPlane.hpp"

#include <iostream>
#include <iomanip>

int main()
{
    float distToOrigin = 10 ; 

    nplane plane = make_nplane( 0,0,1, distToOrigin) ; 

    for(int i=0 ; i < 30 ; i++)
        std::cout << std::setw(4) << i << " " << plane(0.f,0.f,i) << std::endl ;  


    return 0 ; 
}
