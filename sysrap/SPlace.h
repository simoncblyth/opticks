#pragma once

#include "scuda.h"
#include "stran.h"
#include "sstr.h"

#include <glm/glm.hpp>
#include "NP.hh"


struct SPlace
{
    static void AddTransform( double* ttk, const char* opt, const glm::tvec3<double>& a, const glm::tvec3<double>& b, const glm::tvec3<double>& c ); 
    static NP* AroundSphere(   const char* opts, double radius, double item_arclen, unsigned num_ring=10 ); 
    static NP* AroundCylinder( const char* opts, double radius, double halfheight , unsigned num_ring=10, unsigned num_in_ring=16  ); 
};

void SPlace::AddTransform( double* ttk, const char* opt, const glm::tvec3<double>& a, const glm::tvec3<double>& b, const glm::tvec3<double>& c )
{
    if(strcmp(opt,"TR") == 0 || strcmp(opt,"tr") == 0 )
    {
        bool flip = strcmp(opt,"tr") == 0 ; 
        glm::tmat4x4<double> tr = Tran<double>::Place( a, b, c, flip );  
        double* src = glm::value_ptr(tr) ; 
        //std::cout << Tran<double>::Desc(src, 16) << std::endl ; 
        memcpy( ttk, src, 16*sizeof(double) ); 
    }
    else if(strcmp(opt,"R") == 0 || strcmp(opt,"r") == 0)
    {
        bool flip = strcmp(opt,"r") == 0 ; 
        glm::tmat4x4<double> tr = Tran<double>::RotateA2B( a, b, flip );  
        double* src = glm::value_ptr(tr) ; 
        memcpy( ttk, src, 16*sizeof(double) ); 
    }
    else if(strcmp(opt,"T") == 0 || strcmp(opt,"t") == 0)
    {
        bool flip = strcmp(opt,"t") == 0 ; 
        glm::tmat4x4<double> tr = Tran<double>::Translate(c, flip );  
        double* src = glm::value_ptr(tr) ; 
        memcpy( ttk, src, 16*sizeof(double) ); 
    }
    else if(strcmp(opt,"D")== 0)
    {
        for(int l=0 ; l < 3 ; l++) ttk[4*0+l] = a[l] ; 
        for(int l=0 ; l < 3 ; l++) ttk[4*1+l] = b[l] ; 
        for(int l=0 ; l < 3 ; l++) ttk[4*2+l] = c[l] ; 
    }
    else
    {
        std::cout << "SPlace::AddTransform : ERROR opt is not handled [" << opt << "]" << std::endl ; 
    }
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
    NP* transforms(const char* opts) const ; 
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

/**
SPlaceSphere::transforms
---------------------------

The shape of the array returned is (tot_items, num_opts, 4, 4) 
where num_opts depends on the number of comma delimted options
in the opts string. For example "RT,R,T,D" would hav num_opts 4.

**/





NP* SPlace::AroundSphere(const char* opts, double radius, double item_arclen, unsigned num_ring )
{
    SPlaceSphere sp(radius, item_arclen, num_ring); 
    std::cout << sp.desc() << " opts " << opts << std::endl ; 
    return sp.transforms(opts)  ; 
}



NP* SPlaceSphere::transforms(const char* opts) const 
{
    if( tot_item == 0 ) return nullptr ;

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
                double* ttk = tt + item_values*idx+16*k ; 
                const char* opt = vopts[k].c_str();  
                SPlace::AddTransform(ttk, opt, a, b, c ); 
            }    // k:over opt
        }        // j:over items in the ring 
    }            // i:over rings
 
    assert( count == tot_item );  
    return trs ; 
}




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

NP* SPlace::AroundCylinder(const char* opts, double radius, double halfheight, unsigned num_ring, unsigned num_in_ring )
{
    unsigned tot_place = num_ring*num_in_ring ; 

    std::vector<std::string> vopts ; 
    sstr::Split(opts, ',', vopts); 
    unsigned num_opt = vopts.size(); 
    bool dump = false ; 
 
 
    NP* trs = NP::Make<double>( tot_place, num_opt, 4, 4 ); 
    double* tt = trs->values<double>();     
    unsigned item_values = num_opt*4*4 ; 

    
    if(dump) std::cout 
        << "SPlace::AroundCylinder"
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
 
                AddTransform(tt+offset, opt, a, b, c ); 
            }
        }
    }
    assert( count == tot_place ); 
    return trs ; 
}

