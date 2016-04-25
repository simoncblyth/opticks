#include "Demo.hh"

#include "GLMFormat.hpp"

const char* Demo::PREFIX = "demo" ;

const char* Demo::A = "a" ;
const char* Demo::B = "b" ;
const char* Demo::C = "c" ;


const char* Demo::getPrefix()
{    
   return PREFIX ; 
}


std::string Demo::get(const char* name)
{
    float v(0.f) ; 
   
    if(     strcmp(name,A)==0)     v = getA();
    else if(strcmp(name,B)== 0 )   v = getB();
    else if(strcmp(name,C)== 0 )   v = getC();
    else
         printf("Demo::get bad name %s\n", name);

    return gformat(v);
}

void Demo::set(const char* name, std::string& s)
{
    float v = gfloat_(s);

    if(     strcmp(name,A)==0)    setA(v);
    else if(strcmp(name,B)== 0 )  setB(v);
    else if(strcmp(name,C)== 0 )  setC(v);
    else
         printf("Demo::set bad name %s\n", name);
}



