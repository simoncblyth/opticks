#pragma once
#include <vector>
#include <optix.h>
#include <glm/glm.hpp>

#include "IAS.h"
#include "GAS.h"

struct Geo
{
    static Geo* fGeo ; 
    static Geo* Get();  

    Geo(const char* spec_, const char* geometry_);

    void init();
    void init_sphere_containing_grid_of_two_radii_spheres_compound(float& tminf, float& tmaxf);
    void init_sphere_containing_grid_of_two_radii_spheres(float& tminf, float& tmaxf);
    void init_sphere(float& tminf, float& tmaxf);
    void init_sphere_two(float& tminf, float& tmaxf);

    unsigned getNumGAS() const ; 
    unsigned getNumIAS() const ; 

    unsigned getNumBI() const ;
    unsigned getNumBI(unsigned gas_idx) const ;
    unsigned getOffsetBI(unsigned gas_idx) const ;
    void     dumpOffsetBI() const ;

    const GAS& getGAS(int gas_idx_) const ; 
    const IAS& getIAS(int ias_idx_) const ; 

    void makeGAS(float extent);
    void makeGAS(float extent0, float extent1);
    void makeGAS(const std::vector<float>& extents);
    void makeIAS(float extent, float step, const std::vector<unsigned>& gas_modulo, const std::vector<unsigned>& gas_single );


    void writeIAS(unsigned ias_idx, const char* dir) const ;
    void writeGAS(unsigned gas_idx, const char* dir) const ; 
    void write(const char* prefix) const ; 

    static void WriteNP( const char* dir, const char* name, float* data, int ni, int nj, int nk ); 


    AS* getAS(const char* spec) const ;
    void setTop(const char* spec) ; 
    void setTop(AS* top_) ; 
    AS* getTop() const ; 
    float getTopExtent() const ; 

    const char* spec = nullptr ; 
    const char* geometry = nullptr ; 
    AS* top = nullptr ; 

    float tmin = 0.f ; 
    float tmax = 1e16f ; 

    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 
    std::vector<unsigned> nbis ; 

};
