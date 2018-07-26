#include <iostream>
#include "PLOG.hh"
#include "Y4CSG.hh"

Y4CSG::Y4CSG()
{
    std::cout << "ctor START" << std::endl ; 
    LOG(info) << "." ;  
    std::cout << "ctor DONE " << std::endl ; 
}

std::string Y4CSG::desc() const
{
    return "Y4CSG" ; 
}





