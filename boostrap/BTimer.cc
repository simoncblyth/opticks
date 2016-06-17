#include "BTimer.hh"
//#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time ; 

double BTimer::RealTime()
{
    ptime t(microsec_clock::universal_time());
    time_duration d = t.time_of_day();
    double unit = 1e9 ; 
    return d.total_nanoseconds()/unit ;    
}



