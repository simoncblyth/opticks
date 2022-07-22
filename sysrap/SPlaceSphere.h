#pragma once

#include <sstream>
#include <string>
#include <glm/glm.hpp>

#include "scuda.h"
#include "stran.h"
#include "sstr.h"
#include "NP.hh"

#include "SPlaceRing.h"

struct SPlaceSphere
{
    double radius ; 
    double item_arclen ; 
    unsigned num_ring ; 
    SPlaceRing* ring ;
    unsigned tot_item ; 
 
    SPlaceSphere(double radius, double item_arclen, unsigned num_ring); 
    std::string desc() const ; 

    NP* transforms(const char* opts) const ; 
};

inline SPlaceSphere::SPlaceSphere(double radius_, double item_arclen_, unsigned num_ring_)
    :
    radius(radius_),
    item_arclen(item_arclen_),
    num_ring(num_ring_),
    ring(new SPlaceRing[num_ring]),
    tot_item(0)
{
    for(unsigned i=0 ; i < num_ring ; i++) 
    {
        SPlaceRing& rr = ring[i] ; 
        rr.idx  = i ; 
        rr.z    = -radius + 2.*radius*double(i)/double(num_ring-1) ; 
         
        rr.costh = rr.z/radius ; 
        double theta = acos(rr.costh); 

        //rr.sinth = sqrt( 1. - rr.costh*rr.costh ) ;
        rr.sinth = sin(theta) ;
 
        rr.circ = 2.*glm::pi<double>()*radius*rr.sinth  ;
        rr.num  = rr.circ/item_arclen ; 

        tot_item += rr.num ; 
    }
}

inline std::string SPlaceSphere::desc() const 
{
    std::stringstream ss ; 
    ss << SPlaceRing::Desc(ring, num_ring) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



/**
SPlaceSphere::transforms
---------------------------

The shape of the array returned is (tot_items, num_opts, 4, 4) 
where num_opts depends on the number of comma delimted options
in the opts string. For example "RT,R,T,D" would hav num_opts 4.

**/

inline NP* SPlaceSphere::transforms(const char* opts) const 
{
    if( tot_item == 0 ) return nullptr ;

    std::cout << desc() << " opts " << opts << std::endl ; 
    std::vector<std::string> vopts ; 
    sstr::Split(opts, ',', vopts); 
    unsigned num_opt = vopts.size(); 
 
    NP* trs = NP::Make<double>( tot_item, num_opt, 4, 4 ); 
    double* tt = trs->values<double>();     
    unsigned item_values = 4*4*num_opt ; 

    glm::tvec3<double> a(0.,0.,1.); 
    unsigned count = 0 ; 

    for(unsigned i=0 ; i < num_ring ; i++)
    {
        SPlaceRing& rr = ring[i] ; 

        for(unsigned j=0 ; j < rr.num ; j++)
        {
            glm::tvec3<double> u = rr.upoint(j); 
            glm::tvec3<double> b = -u ; 
            glm::tvec3<double> c = radius*u ; 

            unsigned idx = count  ; 
            count += 1 ; 

            for(unsigned k=0 ; k < num_opt ; k++)
            { 
                double* ttk = tt + item_values*idx + 16*k ; 
                const char* opt = vopts[k].c_str();  
                Tran<double>::AddTransform(ttk, opt, a, b, c ); 
            }    // k:over opt
        }        // j:over items in the ring 
    }            // i:over rings
 
    assert( count == tot_item );  
    return trs ; 
}





