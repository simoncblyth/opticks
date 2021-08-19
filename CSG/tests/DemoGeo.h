#pragma once
#include <string>

/**
DemoGeo
=========

DemoGeo is for high level definition of specific examples of geometry, 
that provide tests of the CSG model and rendering thereof.
**/

struct DemoGeo
{
    CSGFoundry*  foundry ; 

    DemoGeo(CSGFoundry* foundry_);

    void init();
    void init_sphere_containing_grid_of_spheres(unsigned layers);
    void init_parade();
    void init_layered(const char* name, float outer, unsigned layers);
    void init_scaled(const char* solid_label_base, const char* demo_node_type, float outer, unsigned layers, unsigned num_gas);
    void init_clustered(const char* name);
    void init(const char* name);

    void addInstance(unsigned gas_idx, float tx=0.f, float ty=0.f, float tz=0.f ) ;

    std::string desc() const ;

};


