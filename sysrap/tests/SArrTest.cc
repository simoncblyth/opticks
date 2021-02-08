#include "SArr.hh"
#include <iostream>

/**
SArrTest
==========

Attempting to make a runtime dynamic type like this doesnt work. 
The template argument must be known at compile time.  
Attemting to do that at runtime gives compilation error::

   non-type template argument is not a constant expression

:google:`C++ runtime dynamically sized type`

**/

int main(int argc, char** argv)
{
    //const unsigned N = argc > 1 ? atoi(argv[1]) : 10 ; // NOPE
    const unsigned N = 10 ; 

    SArr<N>* sa = new SArr<N>() ; 

    for(unsigned i=0 ; i < N ; i++) sa->values[i] = float(i); 
    for(unsigned i=0 ; i < N ; i++) std::cout << sa->values[i] << std::endl ; 

    return 0 ; 
}
