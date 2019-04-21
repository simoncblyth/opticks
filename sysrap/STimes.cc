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
    if( count ==  0_ return "" ; 
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

