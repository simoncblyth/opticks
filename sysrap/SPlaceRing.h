#pragma once

#include <string>
#include <sstream>
#include <glm/glm.hpp>

struct SPlaceRing
{
    unsigned idx ; 
    double z ; 
    double costh ; 
    double sinth ; 
    double circ ; 
    unsigned num ; 

    std::string desc() const ; 
    static std::string Desc(const SPlaceRing* rr, unsigned num_ring); 
    double phi(unsigned item_idx) const ; 
    glm::tvec3<double> upoint(unsigned item_idx); 
};


inline std::string SPlaceRing::desc() const
{
    std::stringstream ss ; 
    ss << "SPlaceRing::desc "
       << " idx "  << std::setw(5) << idx 
       << " z "    << std::fixed << std::setw(15) << std::setprecision(5) << z 
       << " circ " << std::fixed << std::setw(15) << std::setprecision(5) << circ
       << " num "  << std::setw(5) << num
       ; 

    std::string s = ss.str(); 
    return s ; 
}
inline std::string SPlaceRing::Desc(const SPlaceRing* rr, unsigned num_ring)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_ring ; i++ ) ss << rr[i].desc() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}
inline double SPlaceRing::phi(unsigned item_idx) const 
{
    assert( item_idx < num ); 
    return 2.*glm::pi<double>()*double(item_idx)/double(num-1 ) ; 
}

inline glm::tvec3<double> SPlaceRing::upoint(unsigned item_idx )
{
    double ph = phi(item_idx); 

    glm::tvec3<double> pt ; 
    pt.x = sinth*cos(ph) ; 
    pt.y = sinth*sin(ph) ; 
    pt.z = costh ; 
    return pt ; 
}


