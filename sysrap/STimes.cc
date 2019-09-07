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

#include "STimes.hh"

#include <sstream>
#include <cstring>
#include <cstdio>
#include <iomanip>


std::string STimes::brief(const char* msg)
{
    std::stringstream ss ; 
    ss 
       << msg 
       << " vali,comp,prel,lnch "
       << std::fixed << std::setw(7) << std::setprecision(4) 
       << validate 
       << std::fixed << std::setw(7) << std::setprecision(4) 
       << compile 
       << std::fixed << std::setw(7) << std::setprecision(4) 
       << prelaunch
       << std::fixed << std::setw(7) << std::setprecision(4) 
       << launch
       ;
    return ss.str();
}


const char* STimes::description(const char* msg)
{
    if(count == 0) return 0 ; 
    char desc[256];
    snprintf(desc, 256, 
      "%s \n"
      " count %5u   sum \n"
      " validate  %10.4f %10.4f \n"  
      " compile   %10.4f %10.4f \n"  
      " prelaunch %10.4f %10.4f \n"  
      " launch    %10.4f %10.4f \n",
      msg,
      count,
      validate,  validate/count,
      compile ,  compile/count,
      prelaunch, prelaunch/count,
      launch,    launch/count);

    _description = strdup(desc);
    return _description ; 
}



std::string STimes::desc() const 
{
    if( count ==  0) return "" ; 
    std::stringstream ss ; 
    ss 
       << std::setw(11) << ""
       << std::setw(10) << "num"
       << std::setw(10) << "sum"
       << std::setw(10) << "avg"
       << std::endl 

       << std::setw(11) << "validate" 
       << std::setw(10) << count
       << std::fixed << std::setw(10) << std::setprecision(4) << validate 
       << std::fixed << std::setw(10) << std::setprecision(4) << validate/count 
       << std::endl 

       << std::setw(11) << "compile" 
       << std::setw(10) << count
       << std::fixed << std::setw(10) << std::setprecision(4) << compile 
       << std::fixed << std::setw(10) << std::setprecision(4) << compile/count 
       << std::endl 

       << std::setw(11) << "prelaunch" 
       << std::setw(10) << count
       << std::fixed << std::setw(10) << std::setprecision(4) << prelaunch 
       << std::fixed << std::setw(10) << std::setprecision(4) << prelaunch/count 
       << std::endl 

       << std::setw(11) << "launch" 
       << std::setw(10) << count
       << std::fixed << std::setw(10) << std::setprecision(4) << launch 
       << std::fixed << std::setw(10) << std::setprecision(4) << launch/count 
       << std::endl 
       ;

    return ss.str() ; 
}

