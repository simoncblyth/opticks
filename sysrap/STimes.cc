#include "STimes.hh"

#include <cstring>
#include <cstdio>

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
