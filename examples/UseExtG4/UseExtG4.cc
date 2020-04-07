
#include <string>
#include "X4.hh"
#include "OPTICKS_LOG.hh"

void test_Name()
{   
    std::string name = "/dd/material/Water" ;
    
    LOG(info) 
        << std::endl 
        << " name      : " << name   << std::endl   
        << " Name      : " << X4::Name(name)  << std::endl    
        << " ShortName : " << X4::ShortName(name) << std::endl
        << " BaseName  : " << X4::BaseName(name)
        ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    test_Name(); 
    return 0 ; 
}
