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

To present the EpochCount stamps, which now standardize on microseconds (millionths of a second) use::

    In [1]: np.array([1681470877922940,1681470881149639], dtype="datetime64[us]" )
    Out[1]: array(['2023-04-14T11:14:37.922940', '2023-04-14T11:14:41.149639'], dtype='datetime64[us]')

    In [7]: np.array([1681470877922940,1681470881149639], dtype=np.uint64).view("datetime64[us]")
    Out[7]: array(['2023-04-14T11:14:37.922940', '2023-04-14T11:14:41.149639'], dtype='datetime64[us]')

Or without numpy::

    In [1]: from time import localtime, strftime
    In [2]: t = 1681470877922940
    In [3]: print(strftime('%Y-%m-%d %H:%M:%S',localtime(t/1000000)))
    2023-04-14 12:14:37   ## NB +1hr from BST  

**/

#include <cassert>
#include <chrono>
#include <thread>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <cstdint>
#include <sstream>

struct stimer
{
   typedef std::chrono::time_point<std::chrono::system_clock> TP ; 
   typedef std::chrono::duration<double> DT ; 
   enum { UNSET, STARTED, STOPPED } ; 

   static constexpr const char* UNSET_   = "UNSET  " ; 
   static constexpr const char* STARTED_ = "STARTED" ; 
   static constexpr const char* STOPPED_ = "STOPPED" ; 
   static constexpr const char* ERROR_   = "ERROR  " ; 
   static constexpr const char* FORMAT_ZERO_ = "FORMAT_ZERO             " ; 
   static const char* Status(int st) ; 
   static uint64_t EpochCountAsis(const std::chrono::time_point<std::chrono::system_clock>& t0 ); 
   static uint64_t EpochCount(const std::chrono::time_point<std::chrono::system_clock>& t0 ); // microseconds
   static uint64_t EpochCountNow() ;  // microseconds
   static std::chrono::time_point<std::chrono::system_clock> TimePoint( uint64_t epoch_count ); 
   static std::time_t ApproxTime(const std::chrono::time_point<std::chrono::system_clock>& t0 ); 

   static const char* Format(uint64_t epoch_count ); 
   static const char* Format(const std::chrono::time_point<std::chrono::system_clock>& t0 ); 
   static const char* Format(std::time_t tt ); 

   std::string desc() const ; 

   TP _start ; 
   TP _stop ; 
   int status = UNSET ; 

   uint64_t start_count() const ; 
   uint64_t stop_count() const ; 


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

inline uint64_t stimer::EpochCountAsis(const std::chrono::time_point<std::chrono::system_clock>& t0 )
{
   return t0.time_since_epoch().count(); 
}
inline uint64_t stimer::EpochCount(const std::chrono::time_point<std::chrono::system_clock>& t0 )
{
   return std::chrono::duration_cast<std::chrono::microseconds>(t0.time_since_epoch()).count() ;  
} 
inline uint64_t stimer::EpochCountNow()
{
    std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
    return EpochCount(t0); 
}

inline std::chrono::time_point<std::chrono::system_clock> stimer::TimePoint( uint64_t epoch_count_microseconds )
{
    /*
    using clock = std::chrono::system_clock ; 
    clock::duration dur(epoch_count) ; 
    clock::time_point tp(dur);
    */
    std::chrono::system_clock::time_point tp{std::chrono::microseconds{epoch_count_microseconds}};
    return tp ; 
}


inline std::time_t stimer::ApproxTime(const std::chrono::time_point<std::chrono::system_clock>& t0 )
{
    //auto highResNow = std::chrono::system_clock::now();
    //auto systemNow = std::chrono::system_clock::now();
    //auto offset = std::chrono::duration_cast<std::chrono::system_clock::duration>( highResNow - t0 ); 
    //auto output = systemNow + offset  ;
    std::time_t tt =  std::chrono::system_clock::to_time_t( t0 );
    return tt ; 
}
inline const char* stimer::Format(uint64_t epoch_count )
{
    std::chrono::time_point<std::chrono::system_clock> tp = TimePoint(epoch_count); 
    return Format(tp) ; 
}
inline const char* stimer::Format(const std::chrono::time_point<std::chrono::system_clock>& t0 )
{
    return EpochCount(t0) == 0 ? FORMAT_ZERO_ : Format(ApproxTime(t0)) ; 
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
       << " _start " << EpochCount(_start) 
       << " start "  << Format(_start) 
       << " _stop " << EpochCount(_stop) 
       << " stop "   << Format(_stop) 
       << " duration " << std::scientific << duration() 
       ; 
    std::string str = ss.str(); 
    return str ; 
}

inline uint64_t stimer::start_count() const { return EpochCount(_start) ; }
inline uint64_t stimer::stop_count() const {  return EpochCount(_stop) ; }


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
    if(!is_ready()) std::cerr << "stimer::start called when STARTED already ? " << desc() << std::endl ;  
    status = STARTED ; 
    _start = std::chrono::system_clock::now(); 
} 
inline void stimer::stop()
{
    if(!is_started()) std::cerr << "stimer::stop called when not STARTED ? " << desc() << std::endl ;  
    status = STOPPED ; 
    _stop  = std::chrono::system_clock::now(); 
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


