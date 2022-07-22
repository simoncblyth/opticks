#pragma once

#include <glm/glm.hpp>

#include "scuda.h"
#include "stran.h"
#include "sstr.h"
#include "NP.hh"


struct SPlaceCylinder
{
    double radius ; 
    double halfheight ; 
    unsigned num_ring ; 
    unsigned num_in_ring ; 
    unsigned tot_place ; 

    SPlaceCylinder(double radius, double halfheight, unsigned num_ring, unsigned num_in_ring_); 
    std::string desc() const ; 
    NP* transforms(const char* opts) const ; 
};

 
inline SPlaceCylinder::SPlaceCylinder(double radius_, double halfheight_, unsigned num_ring_, unsigned num_in_ring_)
    :
    radius(radius_),
    halfheight(halfheight_),
    num_ring(num_ring_),
    num_in_ring(num_in_ring_),
    tot_place(num_ring*num_in_ring)
{
}


inline std::string SPlaceCylinder::desc() const 
{
    std::stringstream ss ; 
    ss << "SPlaceCylinder::desc" 
       << " radius " << radius 
       << " halfheight " << halfheight 
       << " num_ring " << num_ring 
       << " num_in_ring " << num_in_ring 
       << " tot_place " << tot_place 
       << std::endl 
       ; 
    std::string s = ss.str(); 
    return s ; 
}




inline NP* SPlaceCylinder::transforms(const char* opts) const 
{
    std::vector<std::string> vopts ; 
    sstr::Split(opts, ',', vopts); 
    unsigned num_opt = vopts.size(); 
    bool dump = false ; 
 
    NP* trs = NP::Make<double>( tot_place, num_opt, 4, 4 ); 
    double* tt = trs->values<double>();     
    unsigned item_values = num_opt*4*4 ; 
    
    if(dump) std::cout 
        << "SPlaceCylinder::transforms"
        << " tot_place " << tot_place
        << " num_opt " << num_opt
        << " tt " << tt
        << " item_values " << item_values
        << std::endl 
        ; 

    double zz[num_ring] ; 
    for(unsigned i=0 ; i < num_ring ; i++) zz[i] = -halfheight + 2.*halfheight*double(i)/double(num_ring-1) ; 

    double phi[num_in_ring] ; 
    for(unsigned j=0 ; j < num_in_ring ; j++) phi[j] = glm::pi<double>()*2.*double(j)/double(num_in_ring) ; 
    // not -1 and do not want to have both 0. and 2.pi 

    glm::tvec3<double> a(0.,0.,1.); 

    unsigned count = 0 ; 
    for(unsigned j=0 ; j < num_in_ring ; j++)
    {
        glm::tvec3<double> u(cos(phi[j]),sin(phi[j]),0.); 
        glm::tvec3<double> b = -u ; 
        glm::tvec3<double> c = u*radius ;  

        for(unsigned i=0 ; i < num_ring ; i++)
        {
            c.z = zz[i] ; 
            //unsigned idx = count ; 
            unsigned idx = i*num_in_ring + j ; 

            count += 1 ; 

            for(unsigned k=0 ; k < num_opt ; k++)
            { 
                unsigned offset = item_values*idx+16*k ; 
                const char* opt = vopts[k].c_str();  

                if(dump) std::cout 
                    << "SPlace::AroundCylinder" 
                    << " opt " << opt
                    << " offset " << offset 
                    << std::endl 
                    ;
 
                Tran<double>::AddTransform(tt+offset, opt, a, b, c ); 
            }
        }
    }
    assert( count == tot_place ); 
    return trs ; 
}



