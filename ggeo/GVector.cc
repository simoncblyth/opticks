#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>

#include "GVector.hh"

std::string gfloat3::desc() const 
{
    std::stringstream ss ; 
    ss
          << "(" << std::setw(10) << std::fixed << std::setprecision(3) << x 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << y 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
          << ")"
          ; 

    return ss.str(); 
}

std::string gfloat4::desc() const 
{
    std::stringstream ss ; 
    ss
          << "(" << std::setw(10) << std::fixed << std::setprecision(3) << x 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << y 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << w
          << ")"
          ; 

    return ss.str(); 
}

std::string guint4::description() const 
{
    std::stringstream ss ; 
    unsigned umax = std::numeric_limits<unsigned>::max() ;


    ss << " (" ;

    if(x == umax) ss << "---" ;
    else          ss << std::setw(3) << x ;

    ss << "," ;

    if(y == umax) ss << "---" ;
    else          ss << std::setw(3) << y  ;
            
    ss << "," ;

    if(z == umax) ss << "---" ;
    else          ss << std::setw(3) << z  ;
 
    ss << "," ;

    if(w == umax) ss << "---" ;
    else          ss << std::setw(3) << w  ;
 

    ss << ")" ;


    return ss.str(); 
}



