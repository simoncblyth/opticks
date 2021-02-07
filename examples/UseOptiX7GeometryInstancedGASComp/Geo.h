#pragma once
#include <vector>
#include <optix.h>
#include "IAS.h"
#include "GAS.h"

struct Geo
{
    static Geo* fGeo ; 
    static Geo* Get();  

    Geo(const char* spec_);

    void setTmin(float tmin_); 
    void setTmax(float tmax_); 
    float getTmin() const ; 
    float getTmax() const ; 
 
    void init_sphere_containing_grid_of_two_radii_spheres();
    void init_sphere();
    void init_sphere_two();

    unsigned getNumGAS() const ; 
    unsigned getNumIAS() const ; 

    const GAS& getGAS(int gas_idx_) const ; 
    const IAS& getIAS(int ias_idx_) const ; 

    void makeGAS(float extent);
    void makeGAS(const std::vector<float>& extents);
    void makeIAS(float extent, float step, const std::vector<unsigned>& gas_modulo, const std::vector<unsigned>& gas_single );

    AS* getAS(const char* spec) const ;
    void setTop(const char* spec) ; 
    void setTop(AS* top_) ; 
    AS* getTop() const ; 
    float getTopExtent() const ; 

    const char* spec = nullptr ; 
    AS* top = nullptr ; 
    float tmin = 0.f ; 
    float tmax = 1e16f ; 

    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 

};
