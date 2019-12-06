
// nvcc UseThrust.cu -o /tmp/a.out && /tmp/a.out

#include <sstream>
#include <iostream>
#include <thrust/version.h>


static std::string ThrustVersionString()
{
    std::stringstream ss ;  
    ss 
         <<  THRUST_MAJOR_VERSION 
         << "."
         <<  THRUST_MINOR_VERSION 
         << "."
         <<  THRUST_SUBMINOR_VERSION 
         << "p"
         <<  THRUST_PATCH_NUMBER 
         ;

    return ss.str(); 
}



int main(int argc, char** argv)
{
    std::cout << ThrustVersionString() << std::endl ; 
    return 0 ; 
} 
