#include "NPlane.hpp"
#include "GLMPrint.hpp"
#include <cstring>

void nplane::dump(const char* msg)
{
    print(param, msg);
}

void ndisc::dump(const char* msg)
{
    char dmsg[128];
    snprintf(dmsg, 128, "ndisc radius %10.4f %s \n", radius, msg );
    plane.dump(dmsg);
}

void nbbox::dump(const char* msg)
{
    printf("%s\n", msg);
    print(min, "bb min");
    print(max, "bb max");
}


