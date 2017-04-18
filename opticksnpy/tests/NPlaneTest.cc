#include "NPlane.hpp"

#include "NGLMExt.hpp"

#include <iostream>
#include <iomanip>


void test_sdf()
{
    float distToOrigin = 10 ; 

    nplane plane = make_plane( 0,0,1, distToOrigin) ; 

    for(int i=0 ; i < 30 ; i++)
        std::cout << std::setw(4) << i << " " << plane(0.f,0.f,i) << std::endl ;  
}

void test_intersect()
{
    nplane plane = make_plane( 0,0,1,10) ;

    float tmin = 0.f ; 
    glm::vec3 ray_origin(0,0,0);
    glm::vec3 ray_direction(0,0,0);
 
    for(int i=0 ; i < 2 ; i++)
    {
        switch(i)
        {
           case 0: ray_direction.z = -1 ; break ;
           case 1: ray_direction.z =  1 ; break ;
        }

        glm::vec4 isect ; 
        bool valid_intersect = plane.intersect( tmin, ray_origin, ray_direction, isect );

        std::cout 
            <<  " i " << std::setw(2) << i 
            <<  " ray_origin   " << ray_origin
            <<  " ray_direction " << ray_direction
            <<  " isect " << isect
            <<  " valid_intersect " << valid_intersect 
            << std::endl 
            ; 
    }

}



int main()
{
    test_sdf();
    //test_intersect();


    return 0 ; 
}
