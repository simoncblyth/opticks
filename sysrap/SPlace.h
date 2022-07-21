#pragma once

#include "scuda.h"
#include "stran.h"
#include <glm/glm.hpp>
#include "NP.hh"


struct SPlace
{
    static NP* AroundCylinder( double radius, double halfheight , bool flip=false, unsigned num_ring=10, unsigned num_in_ring=16  ); 
    static NP* AroundSphere(   double radius, double item_arclen, bool flip=false, unsigned num_ring=10 ); 
};

/**
SPlace::AroundCylinder
------------------------

Form placement transforms that orient local-Z axis to 
a radial outwards direction with translation at points around the cylinder.  

flip:false(default)
    transform memory layout has last 4 of 16 elements (actually 12,13,14) 
    holding the translation, which corresponds to the OpenGL standard

flip:true
    transform memory layout has translation in right hand column at elements 3,7,11
    this is needed by pyvista it seems 
    This corresponds to the transposed transform compared with flip:false

**/

NP* SPlace::AroundCylinder(double radius, double halfheight, bool flip, unsigned num_ring, unsigned num_in_ring )
{
    unsigned num_tr = num_ring*num_in_ring ; 

    double zz[num_ring] ; 
    for(unsigned i=0 ; i < num_ring ; i++) zz[i] = -halfheight + 2.*halfheight*double(i)/double(num_ring-1) ; 

    double phi[num_in_ring] ; 
    for(unsigned j=0 ; j < num_in_ring ; j++) phi[j] = glm::pi<double>()*2.*double(j)/double(num_in_ring) ; 
    // not -1 and do not want to have both 0. and 2.pi 

    NP* tr = NP::Make<double>( num_tr, 4, 4 ); 
    double* tt = tr->values<double>(); 
    unsigned item_values = 4*4 ; 

    glm::tvec3<double> a(0.,0.,1.); 

    for(unsigned j=0 ; j < num_in_ring ; j++)
    {
        glm::tvec3<double> u(cos(phi[j]),sin(phi[j]),0.); 
        glm::tvec3<double> b = -u ; 
        glm::tvec3<double> c = u*radius ;  

        for(unsigned i=0 ; i < num_ring ; i++)
        {
            c.z = zz[i] ; 
            glm::tmat4x4<double> t = Tran<double>::Place( a, b, c, flip );  

            //unsigned idx = j*num_ring + i ; 
            unsigned idx = i*num_in_ring + j ; 
 
            double* src = glm::value_ptr(t); 
            double* dst = tt + item_values*idx ; 
            memcpy( dst, src, item_values*sizeof(double) ); 
        }
    }
    return tr ; 
}




/**
SPlace::AroundSphere
-----------------------

The number within each ring at a particular z is determined 
from the circumference the ring at that z. 





            +---+            
        +------------+
      +----------------+
        +------------+
            +---+            


     x = r sin th cos ph
     y = r sin th sin ph       x^2 + y^2 = r^2 sin^2 th
     z = r cos th                    z^2 = r^2 cos^2 th
              
    cos th = z/r 
    sin th = sqrt( 1 - (z/r)^2 )


               r sin th
         +. .+. .+
          \  |  /|
           \ | / | r cos th
            \|/  |
   ---+------+---+---+ 
                 r


Circumference of ring : 2*pi*(r*sin th) = 2*pi*r*sqrt(1 - (z/r)^2 )     
at z=0 -> 2*pi*r at z =r  -> 0  

**/


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


std::string SPlaceRing::desc() const
{
    std::stringstream ss ; 
    ss << "SPlaceRing::desc "
       << " idx " << std::setw(5) << idx 
       << " z " << std::fixed << std::setw(10) << std::setprecision(5) << z 
       << " circ " << std::fixed << std::setw(10) << std::setprecision(5) << circ
       << " num " << std::setw(5) << num
       ; 

    std::string s = ss.str(); 
    return s ; 
}
std::string SPlaceRing::Desc(const SPlaceRing* rr, unsigned num_ring)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_ring ; i++ ) ss << rr[i].desc() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}
double SPlaceRing::phi(unsigned item_idx) const 
{
    assert( item_idx < num ); 
    return 2.*glm::pi<double>()*double(item_idx)/double(num-1 ) ; 
}

glm::tvec3<double> SPlaceRing::upoint(unsigned item_idx )
{
    double ph = phi(item_idx); 

    glm::tvec3<double> pt ; 
    pt.x = sinth*cos(ph) ; 
    pt.y = sinth*sin(ph) ; 
    pt.z = costh ; 
    return pt ; 
}


struct SPlaceSphere
{
    double radius ; 
    double item_arclen ; 
    unsigned num_ring ; 
    SPlaceRing* ring ;
    unsigned tot_item ; 
 
    SPlaceSphere(double radius, double item_arclen, unsigned num_ring); 
    std::string desc() const ; 
    NP* transforms(bool flip, bool dbg_=false) const ; 
};

SPlaceSphere::SPlaceSphere(double radius_, double item_arclen_, unsigned num_ring_)
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

std::string SPlaceSphere::desc() const 
{
    std::stringstream ss ; 
    ss << SPlaceRing::Desc(ring, num_ring) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

NP* SPlaceSphere::transforms(bool flip, bool dbg_) const 
{
    if( tot_item == 0 ) return nullptr ; 

    NP* trs = NP::Make<double>( tot_item, 4, 4 ); 
    double* tt = trs->values<double>();     

    NP* dbg = NP::Make<double>( tot_item, 4, 4 ); 
    double* dd = dbg->values<double>();     

    unsigned item_values = 4*4 ; 

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

            if( dbg_ )
            {
                for(int k=0 ; k < 3 ; k++) dd[item_values*idx+4*0+k] = u[k] ; 
                for(int k=0 ; k < 3 ; k++) dd[item_values*idx+4*1+k] = a[k] ; 
                for(int k=0 ; k < 3 ; k++) dd[item_values*idx+4*2+k] = b[k] ; 
                for(int k=0 ; k < 3 ; k++) dd[item_values*idx+4*3+k] = c[k] ; 
            }
            else
            {
                glm::tmat4x4<double> t = Tran<double>::Place( a, b, c, flip );  
                double* src = glm::value_ptr(t); 
                double* dst = tt + item_values*idx ; 
                memcpy( dst, src, item_values*sizeof(double) ); 
            }

            count += 1 ; 
        }
    }
    assert( count == tot_item );  
    return dbg_ ? dbg : trs ; 
}


NP* SPlace::AroundSphere( double radius, double item_arclen, bool flip, unsigned num_ring )
{
    SPlaceSphere sp(radius, item_arclen, num_ring); 
    std::cout << sp.desc() << std::endl ; 
    return sp.transforms(flip)  ; 
}



