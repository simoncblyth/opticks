#include <iostream>
#include <iomanip>

#include "NCylinder.hpp"

int main()
{
    ncylinder c[4] ; 

    float radius = 200 ;  
    float sizeZ = 400 ; 

    c[0] = make_cylinder(0,0,0,radius,sizeZ,0) ;    // no caps
    c[1] = make_cylinder(0,0,0,radius,sizeZ, CYLINDER_ENDCAP_P ) ;   
    c[2] = make_cylinder(0,0,0,radius,sizeZ, CYLINDER_ENDCAP_Q ) ;    
    c[3] = make_cylinder(0,0,0,radius,sizeZ, CYLINDER_ENDCAP_P | CYLINDER_ENDCAP_Q ) ; // both caps

    for(int i=0 ; i < 4 ; i++) c[i].dump();

    for(float z=-500. ; z <= 500. ; z+=100 )
    {
        std::cout << " z " << std::setw(10) << z ;
        for(int i=0 ; i < 4 ; i++) std::cout << std::setw(10) << c[i](0,0,z) ;
        std::cout << std::endl ;  
    }

    for(float x=-500. ; x <= 500. ; x+=100 )
    {
        std::cout << " x " << std::setw(10) << x ;
        for(int i=0 ; i < 4 ; i++) std::cout << std::setw(10) << c[i](x,0,0) ;
        std::cout << std::endl ;  
    }



    return 0 ; 
}   
