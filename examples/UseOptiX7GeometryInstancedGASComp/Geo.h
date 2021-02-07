#pragma once
#include <vector>
#include <optix.h>
#include "IAS.h"
#include "GAS.h"

struct Geo
{
    static Geo* fGeo ; 
    static Geo* Get();  

    Geo(); 
    void init_sphere_containing_grid_of_two_radii_spheres();

    unsigned getNumGAS() const ; 
    unsigned getNumIAS() const ; 

    const GAS& getGAS(unsigned gas_idx) const ; 
    const IAS& getIAS(unsigned ias_idx) const ; 

    void makeGAS(float extent);
    void makeIAS(float extent, float step, const std::vector<unsigned>& gas_modulo, const std::vector<unsigned>& gas_single );

    AS* getTop() const ; 
    void setTop(AS* top_) ; 

    AS* top = nullptr ; 
    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 

};
