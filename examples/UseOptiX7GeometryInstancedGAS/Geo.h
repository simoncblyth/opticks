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

    OptixTraversableHandle getTop() const ;

    unsigned getNumGAS() const ; 
    const GAS& getGAS(unsigned gas_idx) const ; 
    float getGAS_Extent(unsigned gas_idx) const ;

    unsigned getNumIAS() const ; 
    const IAS& getIAS(unsigned ias_idx) const ; 
    float getIAS_Extent(unsigned ias_idx) const ;

    void makeGAS(float extent);

    void makeIAS(float extent, float step);
    void makeIAS(float extent, float step, const std::vector<unsigned>& gas_modulo, const std::vector<unsigned>& gas_single );

    std::vector<float> vgas_extent ; 
    std::vector<GAS> vgas ; 

    std::vector<float> vias_extent ; 
    std::vector<IAS> vias ; 

};
