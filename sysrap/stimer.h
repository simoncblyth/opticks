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

struct stimer
{
   typedef std::chrono::time_point<std::chrono::high_resolution_clock> TP ; 
   typedef std::chrono::duration<double> DT ; 
   enum { UNSET, STARTED, STOPPED } ; 

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
    assert( is_ready()  ); 
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


