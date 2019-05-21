// TEST=BTimeStampTest om-t

#include "BTimeStamp.hh"

#include <unistd.h>
#include <iostream>

#include <chrono>


int main(int, char** argv)
{
    std::cerr << argv[0] << std::endl ; 

    double t0 = BTimeStamp::RealTime() ; 
    auto s0 = std::chrono::high_resolution_clock::now();    

    sleep(1);
 
    double t1 = BTimeStamp::RealTime() ; 
    auto s1 = std::chrono::high_resolution_clock::now();    

    float dt = t1 - t0 ; 
    std::chrono::duration<double> s10 = s1 - s0;
    double ds = s10.count();     


    std::cerr << t0 << std::endl ; 
    std::cerr << t1 << std::endl ; 
    std::cerr << dt << std::endl ; 
    std::cerr << ds  << std::endl ; 




    return 0 ; 
}
