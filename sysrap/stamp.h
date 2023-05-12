#pragma  once

#include <chrono>

struct stamp
{
    static uint64_t Now(); 
}; 

inline uint64_t stamp::Now()
{
    std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t0.time_since_epoch()).count() ;  
}


