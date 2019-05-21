#include "BTime.hh"

#include <chrono>

#include <boost/chrono/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <iomanip>


void _sleep(unsigned int sleep_milli)
{
    boost::this_thread::sleep(boost::posix_time::millisec(sleep_milli));
}

double check_second_clock(unsigned int sleep_milli)
{
    boost::posix_time::ptime t1 = boost::posix_time::second_clock::local_time();
    _sleep(sleep_milli);
    boost::posix_time::ptime t2 = boost::posix_time::second_clock::local_time();
    boost::posix_time::time_duration dt = t2 - t1;
    double diff = dt.total_milliseconds()/1e6 ; 
    return diff ;
}

double check_microsec_clock(unsigned int sleep_milli)
{

    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
    _sleep(sleep_milli);
    boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration dt = t2 - t1;
    double diff = dt.total_milliseconds()/1e6 ; 
    return diff ;
}


#ifdef __APPLE__
#include <boost/timer/timer.hpp>

double check_timer(unsigned int sleep_milli)
{
    //const boost::timer::nanosecond_type second(1000000000LL);
    const boost::timer::nanosecond_type millisecond(1000000LL);
    boost::timer::cpu_timer timer;
    _sleep(sleep_milli);
    return (timer.elapsed().user + timer.elapsed().system)/millisecond ; 
}

double check_timer2(unsigned int sleep_milli)
{
   boost::timer::cpu_timer timer;
   _sleep(sleep_milli);
   typedef boost::chrono::duration<double> sec_t; // seconds, stored with a double
   sec_t seconds = boost::chrono::nanoseconds(timer.elapsed().user);
   return seconds.count();
}

#endif



int main(int argc, char** argv)
{
    BTime bt ; 
    std::cout << " argc " << argc 
              << " argv[0] " << argv[0]
              << " check " << bt.check()
              << " now " << bt.now("%Y",0)
              << std::endl ; 


   // http://stackoverflow.com/questions/6734375/c-boost-get-current-time-in-milliseconds


   for(unsigned int i=0 ; i < 10 ; i++)
   {
       unsigned int sleep_milli = i*10 ; 

       std::cout << std::setw(5) << i 
                 << " sleep_milli " << std::setw(10) << sleep_milli
                 << " second " << std::setw(10) << check_second_clock(sleep_milli)
                 << " microsec " << std::setw(10) << check_microsec_clock(sleep_milli)
#ifdef __APPLE__
                 << " timer " << std::setw(10) << check_timer(sleep_milli)
                 << " timer2 " << std::setw(10) << check_timer2(sleep_milli)
#endif
                 << std::endl ; 
       
   } 

  
}
