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

    float getExtent(unsigned gas_idx) const ;
    const GAS& getGAS(unsigned gas_idx) const ; 

    unsigned getNumGAS() const ; 
    unsigned getNumIAS() const ; 

    const IAS& getIAS(unsigned ias_idx) const ; 

    void makeGAS(float extent);
    void makeIAS();

    std::vector<float> vextent ; 
    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 

};
