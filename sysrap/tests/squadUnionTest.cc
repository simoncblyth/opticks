
/**

name=squadUnionTest ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

https://stackoverflow.com/questions/252552/why-do-we-need-c-unions

**/

#include "scuda.h"
#include "squad.h"

union qstate2
{
    struct 
    {   
        float m1_refractive_index ; 
        float m1_absorption_length ;
        float m1_scattering_length ; 
        float m1_reemission_prob ; 

        float m2_refractive_index ; 
        float m2_absorption_length ;
        float m2_scattering_length ; 
        float m2_reemission_prob ; 

    } field ;  
        
    quad2 q ;   
};



#include <iostream>

int main()
{
    qstate2 s ; 
    s.q.zero(); 

    s.field.m1_refractive_index = 1.f ; 
    s.field.m2_refractive_index = 1.5f ; 
 
    std::cout << "s.q.q0.f " << s.q.q0.f << std::endl ; 
    std::cout << "s.q.q1.f " << s.q.q1.f << std::endl ; 


    return 0 ; 
}

