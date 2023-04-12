#pragma once
/**
schrono.h
===========

Good for measuring durations, but complicated to 
extract string timestamps : see stime.h for that. 

**/

#include <cstring>
#include <thread>
#include <chrono>
#include <ctime>

namespace schrono
{
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> TP ; 
    typedef std::chrono::duration<double> DT ; 

    inline std::chrono::time_point<std::chrono::high_resolution_clock> stamp()
    {
        TP t = std::chrono::high_resolution_clock::now();
        return t ; 
    }
    inline double duration( 
        std::chrono::time_point<std::chrono::high_resolution_clock>& t0, 
        std::chrono::time_point<std::chrono::high_resolution_clock>& t1 )
    {
        DT _dt = t1 - t0;
        double dt = _dt.count() ;
        return dt ; 
    }

    inline double delta( std::chrono::time_point<std::chrono::high_resolution_clock>& t0 )
    {
        DT _dt = t0.time_since_epoch() ; // HMM: no standard epoch ? so this might be non-absolute
        double dt = _dt.count() ;
        return dt ; 
    }

    inline double delta_stamp()
    {
        TP t0 = stamp(); 
        return delta(t0); 
    }

    inline void sleep(int seconds)
    {
        std::chrono::seconds dura(seconds);
        std::this_thread::sleep_for( dura );
    }

    inline std::time_t approx_time_t(std::chrono::time_point<std::chrono::high_resolution_clock>& t0 )
    {
        auto highResNow = std::chrono::high_resolution_clock::now();
        auto systemNow = std::chrono::system_clock::now();
        auto offset = std::chrono::duration_cast<std::chrono::system_clock::duration>( highResNow - t0 ); 
        auto output = systemNow + offset  ;
        std::time_t tt =  std::chrono::system_clock::to_time_t( output );
        return tt ; 
    }

    inline const char* format( std::time_t tt )
    {
        std::tm* tm = std::localtime(&tt);
        char buffer[32];
        // Format: Mo, 15.06.2009 20:20:00
        std::strftime(buffer, 32, "%a, %d.%m.%Y %H:%M:%S", tm);  
        return strdup(buffer) ; 
     }

    inline const char* format(std::chrono::time_point<std::chrono::high_resolution_clock>& t0)
    {
        std::time_t tt = approx_time_t(t0); 
        return format(tt) ;   
    }


}



