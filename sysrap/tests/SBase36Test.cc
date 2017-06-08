#include "SBase36.hh"
#include "PLOG.hh"

int main(int argc , char** argv )
{
    PLOG_(argc, argv);

    SBase36 b36 ; 

    for(unsigned i=0 ; i < 50 ; i++) 
       std::cout 
          << std::setw(4) << i 
          << " -> "
          << std::setw(4) << b36(i) 
          << std::endl ; 


    return 0 ; 
}
