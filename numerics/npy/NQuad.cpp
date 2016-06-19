#include "NQuad.hpp"





void nuvec4::dump(const char* msg)
{
    printf("%s : %10u %10u %10u %10u \n",msg, x,y,z,w ); 
}
void nivec4::dump(const char* msg)
{
    printf("%s : %10d %10d %10d %10d \n",msg, x,y,z,w ); 
}

void nvec3::dump(const char* msg)
{
    printf("%s : %10.4f %10.4f %10.4f  \n",msg, x,y,z ); 
}

void nvec4::dump(const char* msg)
{
    printf("%s : %10.4f %10.4f %10.4f %10.4f \n",msg, x,y,z,w ); 
}
void nquad::dump(const char* msg)
{
    printf("%s\n", msg);
    f.dump("f");
    u.dump("u");
    i.dump("i");
}


