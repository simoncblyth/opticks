
#include <sstream>
#include <iomanip> 

#include "CThreeVector.hh"

std::string CThreeVector::Format(const G4ThreeVector& vec, int width, int precision  )
{
    std::stringstream ss ; 
    ss 
       << "[ "
       << std::fixed
       << std::setprecision(precision)  
       << std::setw(width) << vec.x()
       << std::setw(width) << vec.y() 
       << std::setw(width) << vec.z() 
       << "] "
       ;
    return ss.str();
}



