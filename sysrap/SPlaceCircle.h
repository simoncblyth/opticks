#pragma once

#include <glm/glm.hpp>

#include "scuda.h"
#include "stran.h"
#include "sstr.h"
#include "NP.hh"


struct SPlaceCircle
{
    static constexpr const double TWOPI = glm::pi<double>()*2.0 ;  
    double radius ; 
    unsigned num_in_ring ; 
    double frac_phase ; 

    SPlaceCircle(double radius, unsigned num_in_ring, double frac_phase ); 
    std::string desc() const ; 

    NP* transforms(const char* opts) const ; 
};

 
inline SPlaceCircle::SPlaceCircle(double radius_, unsigned num_in_ring_, double frac_phase_)
    :
    radius(radius_),
    num_in_ring(num_in_ring_),
    frac_phase(frac_phase_)
{
}

inline std::string SPlaceCircle::desc() const 
{
    std::stringstream ss ; 
    ss << "SPlaceCircle::desc" 
       << " radius " << radius 
       << " num_in_ring " << num_in_ring 
       << " frac_phase " << frac_phase 
       << std::endl 
       ; 
    std::string s = ss.str(); 
    return s ; 
}

/**


               Z
               |
               |
               | 
         ------+-------> X


**/

inline NP* SPlaceCircle::transforms(const char* opts) const 
{
    std::vector<std::string> vopts ; 
    sstr::Split(opts, ',', vopts); 
    unsigned num_opt = vopts.size(); 
    bool dump = false ; 
 
    NP* trs = NP::Make<double>( num_in_ring, num_opt, 4, 4 ); 
    double* tt = trs->values<double>();     
    unsigned item_values = num_opt*4*4 ; 
    
    if(dump) std::cout 
        << "SPlaceCircle::transforms"
        << " num_opt " << num_opt
        << " tt " << tt
        << " item_values " << item_values
        << std::endl 
        ; 

    double phi[num_in_ring] ; 
    for(unsigned j=0 ; j < num_in_ring ; j++) 
    {
        double frac = double(j)/double(num_in_ring) ; 
        phi[j] = TWOPI*(frac_phase + frac ) ; 
    }
    // not num_in_ring-1 as do not want to have both 0. and 2.pi 

    glm::tvec3<double> a(0.,0.,1.);   // +Z axis 

    for(unsigned j=0 ; j < num_in_ring ; j++)
    {
        glm::tvec3<double> u(cos(phi[j]),0, sin(phi[j])); // u: point stepping around unit circle in XZ plane 
        glm::tvec3<double> b = -u ;                       // b: where to rotate the z axis too, inwards  
        glm::tvec3<double> c = u*radius ;                 // c: translation 

        for(unsigned k=0 ; k < num_opt ; k++)
        { 
            unsigned offset = item_values*j+16*k ; 
            const char* opt = vopts[k].c_str();  

            if(dump) std::cout 
                << "SPlace::AroundCircle" 
                << " opt " << opt
                << " offset " << offset 
                << std::endl 
                ;

            Tran<double>::AddTransform(tt+offset, opt, a, b, c );   // rotate a->b and translate by c 
        }
    }
    return trs ; 
}



