#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "Geo.h"

Geo* Geo::fGeo = NULL ; 

Geo::Geo()
{
    fGeo = this ; 
    makeGAS(); 
    makeIAS(); 
}

Geo* Geo::Get()
{
    return fGeo ; 
}

void Geo::makeGAS()
{
    std::vector<float> bb = { -1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f } ;     
    GAS gas = GAS::Build(bb); 
    vgas.push_back(gas); 
}

float unsigned_as_float( unsigned u ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.u = u  ;   
    return uif.f ; 
}

OptixTraversableHandle Geo::getTop() const
{ 
    //return vgas[0].handle;  // OK
    return vias[0].handle ;
}


void Geo::makeIAS()
{
    int n=50 ;   // 
    int s=4 ; 

    std::vector<glm::mat4> trs ; 

    for(int i=-n ; i <= n ; i+=s ){
    for(int j=-n ; j <= n ; j+=s ){
    for(int k=-n ; k <= n ; k+=s ){

        glm::vec3 tlat(i*1.f,j*1.f,k*1.f) ; 
        glm::mat4 tr(1.f) ;
        tr = glm::translate(tr, tlat );
         
        unsigned instance_id = trs.size(); 
        unsigned gas_idx = 0 ; 

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

