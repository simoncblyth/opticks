#include "BTimeStamp.hh"
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time ; 

double BTimeStamp::RealTime()
{
    ptime t(microsec_clock::universal_time());
    time_duration d = t.time_of_day();
    double unit = 1e9 ; 
    return d.total_nanoseconds()/unit ;    
}

double BTimeStamp::RealTime2()
{
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    long microseconds = now.time_of_day().total_microseconds() ;  
    double sec = double(microseconds)/1000000.0;
    return sec ;    
}




