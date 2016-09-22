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
      " count %5u \n"
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
