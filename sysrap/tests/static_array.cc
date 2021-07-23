// name=static_array ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 

#include <string>
#include <iostream>

struct A
{
    static const int NUM_PV = 6 ; 
    static const char* PV[NUM_PV] ; 
    static std::string PVS[NUM_PV] ; 

};

const char* A::PV[NUM_PV] = {
"red",
"green",
"blue",
"cyan",
"magenta",
"yellow"
};  


std::string A::PVS[NUM_PV] = {
"red",
"green",
"blue",
"cyan",
"magenta",
"yellow"
};  


int main(int argc, char** argv)
{
    for(unsigned i=0 ; i < A::NUM_PV ; i++) std::cout << A::PV[i] << std::endl ;  
    std::cout << "----" << std::endl ; 
    for(unsigned i=0 ; i < A::NUM_PV ; i++) std::cout << A::PVS[i] << std::endl ;  
    return 0 ; 
}
