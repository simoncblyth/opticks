// name=union ; gcc $name.cc -I.. -I/usr/local/cuda/include  -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cstring>
#include <iostream>
#include "sutil_vec_math.h"
#include "Quad.h"


struct Test 
{
   union { quad q0 ; char label[16] ; } ;
   quad q1 ;   
};


int main()
{
    Test a ; 
    std::cout << " sizeof(a.label) " << sizeof(a.label) << std::endl ; 

    const char* label = "hell    hell" ; 
    strncpy(a.label,  label, sizeof(a.label) );  

    std::cout 
        << " a.q0.u.x " << a.q0.u.x 
        << " a.q0.u.y " << a.q0.u.y 
        << " a.q0.u.z " << a.q0.u.z 
        << " a.q0.u.w " << a.q0.u.w 
        << std::endl
        ; 

    return 0 ; 
}

