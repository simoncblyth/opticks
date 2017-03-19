#include "NPlane.hpp"
#include "GLMPrint.hpp"
#include <cstring>

void nplane::dump(const char* msg)
{
    param.dump(msg);
}

float ndisc::z() const 
{
   return plane.param.w ;  
}

void ndisc::dump(const char* msg)
{
    char dmsg[128];
    snprintf(dmsg, 128, "ndisc radius %10.4f %s \n", radius, msg );
    plane.dump(dmsg);
}






