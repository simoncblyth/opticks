#include <iostream>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "Geo.h"

Geo* Geo::fGeo = NULL ; 

Geo::Geo()
{
    fGeo = this ; 

    // Three GAS with symmetrical about origin extents  
    //makeGAS(0.5f); 
    //makeGAS(1.0f); 
    makeGAS(1.5f); 

    makeIAS(); 
}

Geo* Geo::Get()
{
    return fGeo ; 
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
    vgas.push_back(gas); 
    vextent.push_back(extent); 
}

unsigned Geo::getNumGAS() const 
{
    assert( vextent.size() == vgas.size() ); 
    return vgas.size() ; 
}
unsigned Geo::getNumIAS() const 
{
    return vias.size() ; 
}
const GAS& Geo::getGAS(unsigned gas_idx) const
{
    assert( gas_idx < vgas.size() ); 
    return vgas[gas_idx] ; 
}
const IAS& Geo::getIAS(unsigned ias_idx) const
{
    assert( ias_idx < vias.size() ); 
    return vias[ias_idx] ; 
}

float Geo::getExtent(unsigned gas_idx) const
{
    assert( gas_idx < vextent.size() ); 
    return vextent[gas_idx] ; 
}



float unsigned_as_float( unsigned u ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.u = u  ;   
    return uif.f ; 
}

OptixTraversableHandle Geo::getTop() const
{ 
    //return vgas[0].handle;  
    return vias[0].handle ;
}

/**
Geo::makeIAS
-------------

Create vector of transfoms and creat IAS from that.
Currently a 3D grid of translate transforms with all available GAS repeated modulo

**/

void Geo::makeIAS()
{
    int ctrl = -1 ; // modulo cycle thru the GAS
    int n=50 ;   // 
    int s=4 ; 

    unsigned num_gas = getNumGAS(); 

    std::cout << "Geo::makeIAS"
              << " num_gas " << num_gas
              << std::endl
              ;

    std::vector<glm::mat4> trs ; 

    for(int i=-n ; i <= n ; i+=s ){
    for(int j=-n ; j <= n ; j+=s ){
    for(int k=-n ; k <= n ; k+=s ){

        glm::vec3 tlat(i*1.f,j*1.f,k*1.f) ; 
        glm::mat4 tr(1.f) ;
        tr = glm::translate(tr, tlat );

        unsigned idx = trs.size(); 
        unsigned instance_id = idx ; 
        unsigned gas_idx = ctrl < 0 ? idx % num_gas : ctrl ; 

        tr[0][3] = unsigned_as_float(instance_id); 
        tr[1][3] = unsigned_as_float(gas_idx) ;
        tr[2][3] = unsigned_as_float(0) ;   
        tr[3][3] = unsigned_as_float(0) ;   

        trs.push_back(tr); 
    }
    }
    }

    IAS ias = IAS::Build(trs); 
    vias.push_back(ias); 
}

