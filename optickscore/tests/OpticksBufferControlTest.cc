#include "OpticksBufferControl.hh"
#include <iostream>
#include <iomanip>

int main()
{
     const char* ctrl = "OPTIX_SETSIZE,OPTIX_INPUT_OUTPUT" ;
     unsigned long long mask = OpticksBufferControl::Parse(ctrl) ;

     std::cout << " ctrl " << ctrl 
               << " mask " << mask 
               << " desc " << OpticksBufferControl::Description(mask)
               << std::endl ; 

     return 0 ; 
}
