#include <sstream>
#include <iomanip>
#include <limits>

#include "GVector.hh"


std::string guint4::description() const 
{
    std::stringstream ss ; 

    //char s[64] ;
    //snprintf(s, 64, " (%3u,%3u,%3u,%3u) ", x, y, z, w);
    //return s ; 

    unsigned umax = std::numeric_limits<unsigned>::max() ;

    ss << " (" << std::setw(3) << x << "," ;

    if(y == umax)
        ss << " - " ;
    else
        ss << std::setw(3) << y  ;
            
    ss << "," ;

    if(z == umax)
        ss << " - " ;
    else
        ss << std::setw(3) << z  ;
 
    ss << "," << std::setw(3) << w << ") " ; 

    return ss.str(); 
}





