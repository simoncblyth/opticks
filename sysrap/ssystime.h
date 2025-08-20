/**
ssystime.h
===========

Following time mechanics of plog

The sstamp.h methods based on uint64_t states and std::chrono
are more flexible than this old school approach, see::

    sstamp::FormatLog  (aka U::FormatLog from np)
    sstamp::Format     (aka U:Formatr from np)

**/

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <cstddef>
#include <sys/time.h>

struct ssystime
{
    time_t time ;
    unsigned short millitm;
    // for ease of persisting the sstamp.h uint64_t approach is better

    ssystime();
    void now();

    template<bool UTC> std::string fmt() const ;

    static std::string utc();
    static std::string local();
};

inline ssystime::ssystime()
   :
   time(0),
   millitm(0)
{
}

inline void ssystime::now()
{
    timeval tv;
    ::gettimeofday(&tv, NULL);

    time = tv.tv_sec ;
    millitm = static_cast<unsigned short>(tv.tv_usec / 1000);
}

template<bool UTC=false>
inline std::string ssystime::fmt() const
{
    tm t;
    if(UTC) ::gmtime_r(&time, &t);
    else    ::localtime_r(&time, &t);

    std::stringstream ss ;
    ss
       << t.tm_year + 1900 << "-"
       << std::setfill('0') << std::setw(2) << t.tm_mon + 1 << "-"
       << std::setfill('0') << std::setw(2) << t.tm_mday    << " "
       << std::setfill('0') << std::setw(2) << t.tm_hour    << ":"
       << std::setfill('0') << std::setw(2) << t.tm_min     << ":"
       << std::setfill('0') << std::setw(2) << t.tm_sec     << "."
       << std::setfill('0') << std::setw(3) << static_cast<int> (millitm)
       << " "
       ;

    std::string str = ss.str() ;
    return str ;
}

inline std::string ssystime::utc()
{
    ssystime t ;
    t.now();
    return t.fmt<true>() ;
}

inline std::string ssystime::local()
{
    ssystime t ;
    t.now();
    return t.fmt<false>() ;
}

inline std::ostream& operator<<(std::ostream& os, const ssystime& t)
{
    os << t.fmt<false>() ;
    return os;
}

