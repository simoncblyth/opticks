#include "Nuv.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>

std::string nuv::desc() const 
{
    std::stringstream ss ; 

    ss << "(" 
       << std::setw(1) << s()
       << ";"
       << std::setw(2) << u() 
       << ","
       << std::setw(2) << v()
       << ")"
       ;

    return ss.str();
}


std::string nuv::detail() const 
{
    std::stringstream ss ; 

    ss << "nuv "
       << " s " << std::setw(1) << s()
       << " u " << std::setw(3) << u() << "/" << std::setw(3) << nu() 
       << " v " << std::setw(3) << v() << "/" << std::setw(3) << nv() 
       ;

    return ss.str();
}



