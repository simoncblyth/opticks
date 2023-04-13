#pragma once
/**
stimer.h
==========

Usage::

    stimer* t = new stimer  ; 
    t->start(); 
    t->stop(); 
    double dt = t->duration(); // duration between start and stop 


    stimer* t = stimer::create() ; 
    double dt0 = t->lap() ;   // duration between the create and the lap call
    double dt1 = t->lap() ;   // duration between this call and the last 

**/

#include <cassert>
#include <chrono>
#include <thread>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <sstream>

struct stimer
{
   typedef std::chrono::time_point<std::chrono::high_resolution_clock> TP ; 
   typedef std::chrono::duration<double> DT ; 
   enum { UNSET, STARTED, STOPPED } ; 

   static constexpr const char* UNSET_   = "UNSET  " ; 
   static constexpr const char* STARTED_ = "STARTED" ; 
   static constexpr const char* STOPPED_ = "STOPPED" ; 
   static constexpr const char* ERROR_   = "ERROR  " ; 
   static constexpr const char* FORMAT_ZERO_ = "FORMAT_ZERO             " ; 
   static const char* Status(int st) ; 
   static double EpochCount(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0 ); 
   static std::time_t ApproxTime(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0 ); 
   static const char* Format(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0 ); 
   static const char* Format( std::time_t tt ); 

   std::string desc() const ; 


   TP _start ; 
   TP _stop ; 
   int status = UNSET ; 

   // higher level API
   static stimer* create() ; 
   double done(); 
   double lap(); 

   // low level API
   bool is_ready() const ; 
   bool is_started() const ; 
   bool is_stopped() const ; 

   void start(); 
   void stop(); 
   double duration() const ; 

   // for testing 
   static void sleep(int seconds) ; 
};


inline const char* stimer::Status(int st)
{
    const char* str = nullptr ; 
    switch(st)
    {
        case UNSET: str = UNSET_     ; break ; 
        case STARTED: str = STARTED_ ; break ; 
        case STOPPED: str = STOPPED_ ; break ;
        default:      str = ERROR_   ; break ;  
    }
    return str ; 
}

inline double stimer::EpochCount(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0 )
{
   return t0.time_since_epoch().count(); 
}

inline std::time_t stimer::ApproxTime(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0 )
{
    auto highResNow = std::chrono::high_resolution_clock::now();
    auto systemNow = std::chrono::system_clock::now();
    auto offset = std::chrono::duration_cast<std::chrono::system_clock::duration>( highResNow - t0 ); 
    auto output = systemNow + offset  ;
    std::time_t tt =  std::chrono::system_clock::to_time_t( output );
    return tt ; 
}
inline const char* stimer::Format(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0 )
{
    return EpochCount(t0) == 0. ? FORMAT_ZERO_ : Format(ApproxTime(t0)) ; 
}

inline const char* stimer::Format( std::time_t tt )
{
    std::tm* tm = std::localtime(&tt);
    char buffer[32];
    // Format: Mo, 15.06.2009 20:20:00
    std::strftime(buffer, 32, "%a, %d.%m.%Y %H:%M:%S", tm);  
    return strdup(buffer) ; 
}
inline std::string stimer::desc() const 
{
    std::stringstream ss ; 
    ss << "stimer::desc"
       << " status " << Status(status)
       << " start "  << Format(_start) 
       << " stop "   << Format(_stop) 
       << " duration " << std::scientific << duration() 
       ; 
    std::string str = ss.str(); 
    return str ; 
}


inline stimer* stimer::create()
{
    stimer* t = new stimer ; 
    t->start(); 
    return t ; 
}
inline double stimer::done() 
{
    stop(); 
    return duration(); 
}
inline double stimer::lap() 
{
    stop(); 
    double dt = duration(); 
    start(); 
    return dt ; 
}


inline bool stimer::is_ready() const {   return status == UNSET || status == STOPPED ; }
inline bool stimer::is_started() const { return status == STARTED ; }
inline bool stimer::is_stopped() const { return status == STOPPED ; }

inline void stimer::start()
{
    if(!is_ready()) std::cerr << "stimer::start starting again ? " << desc() << std::endl ;  
    status = STARTED ; 
    _start = std::chrono::high_resolution_clock::now(); 
} 
inline void stimer::stop()
{
    assert( is_started() ); 
    status = STOPPED ; 
    _stop  = std::chrono::high_resolution_clock::now(); 
} 

inline double stimer::duration() const 
{
    if(!is_stopped()) return -1. ;   
    DT _dt = _stop - _start ; 
    return _dt.count(); 
}



inline void stimer::sleep(int seconds) // static
{
    std::chrono::seconds dura(seconds);
    std::this_thread::sleep_for( dura );
}


