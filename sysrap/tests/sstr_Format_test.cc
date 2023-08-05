// name=sstr_Format_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "ssys.h"
#include "sstr.h"

void test_0()
{
    std::string path = sstr::Format_("$FOLD/sip_%s.npy", ssys::getenvvar("MOI", "Hama:0:1000", ':', '_' ));   
    std::cout << " path : " << path << std::endl; 
}

void test_1()
{
    const char* omat = "Pyrex" ; 
    const char* osur = "OSUR" ; 
    const char* isur = "ISUR" ; 
    const char* imat = "Vacuum" ; 

    std::string bnd = sstr::Format_("%s/%s/%s/%s", omat, osur, isur, imat ); 
    std::cout << " bnd : " << bnd << std::endl; 
}
void test_Join()
{
    const char* omat = "Pyrex" ; 
    const char* osur = nullptr ; 
    const char* isur = nullptr ; 
    const char* imat = "Vacuum" ; 

    std::string bnd = sstr::Join("/", omat, osur, isur, imat ); 
    std::cout << " bnd : " << bnd << std::endl; 
}

int main(int argc, char** argv)
{
    test_1(); 
    test_Join(); 
    return 0 ; 
}
