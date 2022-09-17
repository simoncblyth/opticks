// name=ssys_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <string>
#include <iostream>
#include "ssys.h"

void test_popen_0()
{
    const char* cmd = "md5 -q -s hello" ; 
    bool chomp = false ; 
    std::string ret = ssys::popen(cmd, chomp); 

    std::cout 
        << " cmd [" << cmd << "]" 
        << std::endl 
        << " ret [" << ret << "]" 
        << std::endl 
        ; 
}

void test_popen_1()
{
    std::cout << ssys::popen("md5 -q -s hello") << std::endl ; 
}

void test_getenvv()
{
    int i = ssys::getenvv("i",0) ; 
    unsigned u = ssys::getenvv("u",0) ; 
    float f = ssys::getenvv("f",0.f) ; 
    double d = ssys::getenvv("d",0.) ; 

    std::cout 
        << "i " << i << std::endl 
        << "u " << u << std::endl 
        << "f " << f << std::endl 
        << "d " << d << std::endl 
        ; 
}


int main(int argc, char** argv)
{
    /*
    test_popen_0(); 
    test_popen_1();
    */

    test_getenvv(); 
 
    return 0 ; 
}
