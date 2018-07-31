// TEST=ArrayTest om-t 

#include "OPTICKS_LOG.hh"

#include <sstream>
#include <string>
#include <cstring>
#include <array>

struct Demo
{
    Demo(const char* name_, int value_)
       :
       name(strdup(name_)),
       value(value_) 
    {
    }
    std::string desc() const 
    {
        std::stringstream ss ; 
        ss
           << std::setw(10) << name
           << " : " 
           << std::setw(10) << value
           ;
        return ss.str();
    }

    const char* name ; 
    int value ; 
};


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    std::array<Demo*, 10> arr ; 
    arr.fill(NULL); 

    arr[5] = new Demo("yo", 42) ; 

    for(int i=0 ; i < arr.size() ; i++) LOG(info) << i << " : " << ( arr[i] ? arr[i]->desc() : "-" ) ; 

    arr[10] = new Demo("hmm", 42) ; 

    return 0 ; 
}


