#include <iostream>
#include "PLOG.hh"
#include "X4CSG.hh"

X4CSG::X4CSG()
{
    std::cout << "ctor START" << std::endl ; 
    LOG(info) << "." ;  
    std::cout << "ctor DONE " << std::endl ; 
}

std::string X4CSG::desc() const
{
    return "X4CSG" ; 
}





