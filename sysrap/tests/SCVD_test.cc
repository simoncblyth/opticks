// name=SCVD_test ; gcc $name.cc -std=c++11 -I.. -lstdc++ -o /tmp/$name && CVD=0,1 /tmp/$name

#include "SCVD.hh"

int main()
{
    SCVD::ConfigureVisibleDevices(); 

    std::cout << "[" << SCVD::Label() << "]" << std::endl ; 

    return 0 ; 
}


