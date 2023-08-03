#pragma  once

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

struct stamp
{
    static uint64_t Now(); 
    static std::string Format(uint64_t t=0, const char* fmt="%FT%T."); 
}; 

inline uint64_t stamp::Now()
{
    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds ;
    std::chrono::time_point<Clock> t0 = Clock::now();
    return std::chrono::duration_cast<Unit>(t0.time_since_epoch()).count() ;  
}
/**
stamp::Format
--------------

Time string from uint64_t with the microseconds since UTC epoch, 
t=0 is special cased to give the current time

**/

inline std::string stamp::Format(uint64_t t, const char* fmt)
{
    if(t == 0) t = Now() ; 
    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds  ; 
    std::chrono::time_point<Clock> tp{Unit{t}} ; 

    std::time_t tt = Clock::to_time_t(tp);

    // extract the sub second part from the duration since epoch
    auto subsec = std::chrono::duration_cast<Unit>(tp.time_since_epoch()) % std::chrono::seconds{1};

    std::stringstream ss ; 
    ss 
       << std::put_time(std::localtime(&tt), fmt ) 
       << std::setfill('0') 
       << std::setw(6) << subsec.count() 
       ;

    std::string str = ss.str(); 
    return str ; 
}

