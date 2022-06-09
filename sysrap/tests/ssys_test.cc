// name=ssys_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <string>
#include <iostream>
#include "ssys.h"

void test_0()
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

void test_1()
{
    std::cout << ssys::popen("md5 -q -s hello") << std::endl ; 
}

int main(int argc, char** argv)
{
    test_0(); 
    test_1(); 
    return 0 ; 
}
