#pragma once
#include <vector>
#include <optix.h>
#include "IAS.h"
#include "GAS.h"

struct Geo
{
    Geo(); 

    static Geo* fGeo ; 
    static Geo* Get();  

    OptixTraversableHandle getTop() const ;

    unsigned getNumGAS() const ; 
    const GAS& getGAS(unsigned gas_idx) const ; 
    float getGAS_Extent(unsigned gas_idx) const ;

    unsigned getNumIAS() const ; 
    const IAS& getIAS(unsigned ias_idx) const ; 
    float getIAS_Extent(unsigned ias_idx) const ;

    void makeGAS(float extent);
    void makeIAS(float extent, float step);

    std::vector<float> vgas_extent ; 
    std::vector<GAS> vgas ; 

    std::vector<float> vias_extent ; 
    std::vector<IAS> vias ; 

};
