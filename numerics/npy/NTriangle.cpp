#include "NTriangle.hpp"

#include "GLMPrint.hpp"
#include "NLog.hpp"

void ntriangle::dump(const char* msg)
{
    LOG(info) << msg ; 
    print(p[0], "p[0]");
    print(p[1], "p[1]");
    print(p[2], "p[2]");
}



