#include "BTimer.hh"
#include <iostream>

int main(int, char** argv)
{
    std::cerr << argv[0] 
              << " BTimer::RealTime " << BTimer::RealTime() 
              << std::endl ; 

    return 0 ; 
}
