// name=sstr_Format_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "ssys.h"
#include "sstr.h"

int main(int argc, char** argv)
{
    std::string path = sstr::Format_("$FOLD/sip_%s.npy", ssys::getenvvar("MOI", "Hama:0:1000", ':', '_' ));   
    std::cout << " path : " << path << std::endl; 
    return 0 ; 
}
