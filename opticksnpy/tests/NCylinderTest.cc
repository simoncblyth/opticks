#include <iostream>
#include <iomanip>

#include "NCylinder.hpp"

int main()
{
    unsigned flags = 0 ; 
    ncylinder c = make_ncylinder(0,0,0,200,400,flags) ;   
    c.pdump();


    for(float z=-500. ; z <= 500. ; z+=100 )
        std::cout << " z " << std::setw(5) << z 
                  << " df " << std::setw(8) << c(0,0,z) 
                  << std::endl ; 

    for(float x=-500. ; x <= 500. ; x+=100 )
        std::cout << " x " << std::setw(5) << x 
                  << " df " << std::setw(8) << c(x,0,0) 
                  << std::endl ; 






    return 0 ; 
}   
