
#include <iostream>
#include <cstring>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "Geo.h"

Geo* Geo::fGeo = NULL ; 

Geo::Geo(const char* spec_) 
   :
   spec(strdup(spec_))  
{
    fGeo = this ; 

    init_sphere_containing_grid_of_two_radii_spheres();
    //init_sphere();
    //init_sphere_two();
}

void Geo::init_sphere_containing_grid_of_two_radii_spheres()
{
    std::cout << "Geo::init_sphere_containing_grid_of_two_radii_spheres " << spec << std::endl ; 

    float ias_extent = 10.f ; 
    float ias_step = 2.f ; 

    makeGAS(0.7f); 
    makeGAS(1.0f); 
    std::vector<unsigned> gas_modulo = {0, 1} ;

    makeGAS(ias_extent*2.0f); 
    std::vector<unsigned> gas_single = {2} ;

    makeIAS(ias_extent, ias_step, gas_modulo, gas_single ); 

    setTop(spec); 

    float top_extent = getTopExtent(); 
    setTmin(top_extent*0.75f) ;   // <-- so can see inside the big sphere  
    setTmax(top_extent*10000.f) ; 
}
void Geo::init_sphere()
{
    std::cout << "Geo::init_sphere" << std::endl ; 
    makeGAS(100.f); 
    setTop("g0"); 

    float top_extent = getTopExtent(); 
    std::cout << "Geo::init_sphere top_extent " << top_extent  << std::endl ; 

    setTmin(top_extent*1.60f) ;   //  hmm depends on viewpoint, aiming to cut into the sphere with the tmin
    setTmax(top_extent*10000.f) ; 
}
void Geo::init_sphere_two()
{
    std::cout << "Geo::init_sphere_two" << std::endl ; 
    std::vector<float> extents = {100.f, 99.f, 98.f, 97.f, 96.f } ; 
    makeGAS(extents); 
    setTop("g0"); 

    float top_extent = getTopExtent(); 
    std::cout << "Geo::init_sphere_two top_extent " << top_extent  << std::endl ; 

    setTmin(top_extent*1.50f) ;   //  hmm depends on viewpoint, aiming to cut into the sphere with the tmin
    setTmax(top_extent*10000.f) ; 
}


void Geo::setTmin(float tmin_)
{
    tmin = tmin_ ; 
}
float Geo::getTmin() const 
{
    return tmin ; 
}

void Geo::setTmax(float tmax_)
{
    tmax = tmax_ ; 
}
float Geo::getTmax() const 
{
    return tmax ; 
}






Geo* Geo::Get()
{
    return fGeo ; 
}

AS* Geo::getTop() const 
{
    return top ; 
}

float Geo::getTopExtent() const 
{
    assert(top); 
    return top ? top->extent : -1.f ;  
}



void Geo::setTop(const char* spec)
{
    AS* a = getAS(spec); 
    setTop(a); 



}

AS* Geo::getAS(const char* spec) const 
{
   assert( strlen(spec) > 1 ); 
   char c = spec[0]; 
   assert( c == 'i' || c == 'g' ); 
   int idx = atoi( spec + 1 ); 

   std::cout << "Geo::getAS " << spec << " c " << c << " idx " << idx << std::endl ; 

   AS* a = nullptr ; 
   if( c == 'i' )
   {
       const IAS& ias = getIAS(idx); 
       a = (AS*)&ias ; 
   } 
   else if( c == 'g' )
   {
       const GAS& gas = getGAS(idx); 
       a = (AS*)&gas ; 
   }

   if(a)
   {
       std::cout << "Geo::getAS " << spec << " a->extent " << a->extent << std::endl ; 
   } 
   return a ; 
}



void Geo::setTop(AS* top_)
{
    top = top_ ; 




}



/**
Geo::makeGAS
---------------

GAS can hold multiple bbox, but for now use just use one each 
with symmetric extent about the origin.

**/
void Geo::makeGAS(float extent)
{
    std::cout << "Geo::makeGAS extent " << extent << std::endl ; 
    std::vector<float> bb = { -extent, -extent, -extent, +extent, +extent, +extent } ;  

    GAS gas = GAS::Build(bb); 
    gas.extent = extent ; 
    vgas.push_back(gas); 
}


/**
Geo::makeGAS
--------------

The first extent must be the largest 


**/
void Geo::makeGAS(const std::vector<float>& extents)
{
    std::cout << "Geo::makeGAS extents.size() " << extents.size() << std::endl ; 
    std::vector<float> bb ; 

    float extent0 = extents[0] ; 

    for(unsigned i=0 ; i < extents.size() ; i++)
    {
        float extent = extents[i] ;
        assert( extent <= extent0 );  
 
        bb.push_back(-extent); 
        bb.push_back(-extent); 
        bb.push_back(-extent); 
        bb.push_back(+extent); 
        bb.push_back(+extent); 
        bb.push_back(+extent); 
    }

    GAS gas = GAS::Build(bb); 
    gas.extent = extent0 ; 
    vgas.push_back(gas); 
}






unsigned Geo::getNumGAS() const 
{
    return vgas.size() ; 
}
unsigned Geo::getNumIAS() const 
{
    return vias.size() ; 
}
const GAS& Geo::getGAS(int gas_idx_) const
{
    unsigned gas_idx = gas_idx_ < 0 ? vgas.size() + gas_idx_ : gas_idx_ ;  
    assert( gas_idx < vgas.size() ); 
    return vgas[gas_idx] ; 
}
const IAS& Geo::getIAS(int ias_idx_) const
{
    unsigned ias_idx = ias_idx_ < 0 ? vias.size() + ias_idx_ : ias_idx_ ;  
    assert( ias_idx < vias.size() ); 
    return vias[ias_idx] ; 
}



float unsigned_as_float( unsigned u ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.u = u  ;   
    return uif.f ; 
}


/**
Geo::makeIAS
-------------

Create vector of transfoms and creat IAS from that.
Currently a 3D grid of translate transforms with all available GAS repeated modulo

**/

void Geo::makeIAS(float extent, float step, const std::vector<unsigned>& gas_modulo, const std::vector<unsigned>& gas_single )
{
    int n=int(extent) ;   // 
    int s=int(step) ; 

    unsigned num_gas = getNumGAS(); 
    unsigned num_gas_modulo = gas_modulo.size() ; 
    unsigned num_gas_single = gas_single.size() ; 


    std::cout << "Geo::makeIAS"
              << " num_gas " << num_gas
              << " num_gas_modulo " << num_gas_modulo
              << " num_gas_single " << num_gas_single
              << std::endl
              ;

    for(unsigned i=0 ; i < num_gas_modulo ; i++ ) assert(gas_modulo[i] < num_gas) ; 
    for(unsigned i=0 ; i < num_gas_single ; i++ ) assert(gas_single[i] < num_gas) ; 


    std::vector<glm::mat4> trs ; 

    for(int i=0 ; i < int(num_gas_single) ; i++)
    {
        unsigned idx = trs.size(); 
        unsigned instance_id = idx ; 
        unsigned gas_idx = gas_single[i] ; 

        glm::mat4 tr(1.f) ;
        tr[0][3] = unsigned_as_float(instance_id); 
        tr[1][3] = unsigned_as_float(gas_idx) ;
        tr[2][3] = unsigned_as_float(0) ;   
        tr[3][3] = unsigned_as_float(0) ;   

        trs.push_back(tr); 
    }


    for(int i=-n ; i <= n ; i+=s ){
    for(int j=-n ; j <= n ; j+=s ){
    for(int k=-n ; k <= n ; k+=s ){

        glm::vec3 tlat(i*1.f,j*1.f,k*1.f) ; 
        glm::mat4 tr(1.f) ;
        tr = glm::translate(tr, tlat );

        unsigned idx = trs.size(); 
        unsigned instance_id = idx ; 
        unsigned gas_modulo_idx = idx % num_gas_modulo ; 
        unsigned gas_idx = gas_modulo[gas_modulo_idx] ; 

        tr[0][3] = unsigned_as_float(instance_id); 
        tr[1][3] = unsigned_as_float(gas_idx) ;
        tr[2][3] = unsigned_as_float(0) ;   
        tr[3][3] = unsigned_as_float(0) ;   

        trs.push_back(tr); 
    }
    }
    }

    IAS ias = IAS::Build(trs); 
    ias.extent = extent ; 
    vias.push_back(ias); 
}

