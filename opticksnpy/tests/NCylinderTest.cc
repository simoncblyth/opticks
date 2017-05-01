#include <iostream>
#include <iomanip>

#include "NCylinder.hpp"

int main()
{
    enum { nc = 3 }; 
    ncylinder c[nc] ; 

    float radius = 200 ;  
    float z1 = 0 ; 
    float z2 = 400 ; 

    c[0] = make_cylinder(radius,z1,z2) ;   
    c[1] = make_cylinder(radius,z1+100.f,z2+100.f) ;   
    c[2] = make_cylinder(radius,z1-100.f,z2-100.f) ;   

    for(int i=0 ; i < nc ; i++) c[i].dump();

    for(float z=-500. ; z <= 500. ; z+=100 )
    {
        std::cout << " z " << std::setw(10) << z ;
        for(int i=0 ; i < nc ; i++) std::cout << std::setw(10) << c[i](0,0,z) ;
        std::cout << std::endl ;  
    }

    for(float x=-500. ; x <= 500. ; x+=100 )
    {
        std::cout << " x " << std::setw(10) << x ;
        for(int i=0 ; i < nc ; i++) std::cout << std::setw(10) << c[i](x,0,0) ;
        std::cout << std::endl ;  
    }



    return 0 ; 
}   
