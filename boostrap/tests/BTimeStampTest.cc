#include "BTimeStamp.hh"
#include <iostream>

int main(int, char** argv)
{
    std::cerr << argv[0] 
              << " BTimeStamp::RealTime " << BTimeStamp::RealTime() 
              << std::endl ; 

    return 0 ; 
}
