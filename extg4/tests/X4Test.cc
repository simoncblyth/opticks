// TEST=X4Test om-t

#include <string>
#include "X4.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv);
 
    std::string name = "/dd/material/Water" ; 

    LOG(info) 
        << std::endl 
        << " name      : " << name   << std::endl   
        << " Name      : " << X4::Name(name)  << std::endl    
        << " ShortName : " << X4::ShortName(name) << std::endl     
        << " BaseName  : " << X4::BaseName(name)     
        ;


    return 0 ; 
}


