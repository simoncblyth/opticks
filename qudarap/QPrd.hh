#pragma once

#include "QUDARAP_API_EXPORT.hh"
#include <vector>
struct float4 ; 
struct quad2 ; 
struct QBnd ; 

struct QUDARAP_API QPrd
{
    const QBnd* bnd ; 

    std::vector<unsigned> bnd_idx ; 
    std::vector<float4> nrmt ; 
    std::vector<quad2>  prd ; 

    QPrd(const QBnd* bnd); 
    void init();   
    void dump(const char* msg="QPrd::dump") const ; 

};



