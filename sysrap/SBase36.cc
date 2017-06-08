#include <sstream>
#include "SBase36.hh"

const char* SBase36::LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" ;

std::string SBase36::operator()( unsigned int val ) const 
{
    //  https://en.wikipedia.org/wiki/Base36
    std::stringstream ss ; 
    do { ss << LABELS[val % 36] ; } while (val /= 36);
    std::string r = ss.str(); 
    std::string rr(r.rbegin(), r.rend());
    return rr ;
}   



