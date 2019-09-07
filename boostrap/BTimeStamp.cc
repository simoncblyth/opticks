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




