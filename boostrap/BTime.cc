/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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




