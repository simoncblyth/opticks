
#include "BTime.hh"

#include <string>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

using namespace boost::gregorian;
using namespace boost::posix_time;


BTime::BTime()
{
    std::cout << "BTime::BTime BRAP_API " << std::endl ; 
}


int BTime::check()
{
   return 42 ; 
}



void BTime::current_time(std::string& ct,  const char* tfmt, int utc)
{
   /*
   time_t t;
   time (&t); 
   struct tm* tt = utc ? gmtime(&t) : localtime(&t) ;
   strftime(buf, buflen, tfmt, tt);
   */


   if(utc) std::cout << "dummy access to utc" << std::endl  ; 

   time_facet* facet = new time_facet(tfmt);

   std::stringstream ss ;

   std::locale loc(ss.getloc(), facet);

   ss.imbue(loc);

   ss << second_clock::local_time() ;

   std::string s = ss.str();

   ct.assign(s) ;

}


std::string BTime::now(const char* tfmt,  int utc )
{
    std::string s ;
    current_time( s, tfmt, utc );  
    return s ; 
}




