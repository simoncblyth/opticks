// name=d4_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>

int main(int argc, char** argv)
{
    struct { double x,y,z,w ; } d4 ; 

    d4.x = 0. ; 
    d4.y = 1. ; 
    d4.z = 2. ; 
    d4.w = 3. ; 

    std::cout 
        << " d4.x " << d4.x 
        << " d4.y " << d4.y 
        << " d4.z " << d4.z 
        << " d4.w " << d4.w 
        << std::endl 
        ; 


    return 0 ; 
}
