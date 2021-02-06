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
    void makeGAS();
    void makeIAS();

    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 

};
