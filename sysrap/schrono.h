#pragma once
/**
schrono.h
===========

Good for measuring durations, but complicated to 
extract string timestamps : see stime.h for that. 

**/

#include <thread>
#include <chrono>
#include <ctime>

namespace schrono
{
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> TP ; 
    typedef std::chrono::duration<double> DT ; 

    std::chrono::time_point<std::chrono::high_resolution_clock> stamp()
    {
        TP t = std::chrono::high_resolution_clock::now();
        return t ; 
    }
    double duration( 
        std::chrono::time_point<std::chrono::high_resolution_clock>& t0, 
        std::chrono::time_point<std::chrono::high_resolution_clock>& t1 )
    {
        DT _dt = t1 - t0;
        double dt = _dt.count() ;
        return dt ; 
    }

    void sleep(int seconds)
    {
        std::chrono::seconds dura(seconds);
        std::this_thread::sleep_for( dura );
    }

    std::time_t approx_time_t(std::chrono::time_point<std::chrono::high_resolution_clock>& t0 )
    {
        auto highResNow = std::chrono::high_resolution_clock::now();
        auto systemNow = std::chrono::system_clock::now();
        auto offset = std::chrono::duration_cast<std::chrono::system_clock::duration>( highResNow - t0 ); 
        auto output = systemNow + offset  ;
        std::time_t tt =  std::chrono::system_clock::to_time_t( output );
        return tt ; 
    }

}



