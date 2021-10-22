#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>


#include <vector_types.h>

#include "SStr.hh"
#include "SSys.hh"
#include "NP.hh"
#include "PLOG.hh"

#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "CSGPrim.h"

#include "DemoGeo.h"
#include "DemoGrid.h"

DemoGeo::DemoGeo(CSGFoundry* foundry_)
    :
    foundry(foundry_),
    maker(foundry->maker)
{
    init();
}

void DemoGeo::init()
{
    LOG(info) << "[" ; 

    const char* geometry = SSys::getenvvar("GEOMETRY", "parade" ); 
    float outer = SSys::getenvint("OUTER", 100.f ) ; 
    int layers = SSys::getenvint("LAYERS", 1) ; 
    int numgas = SSys::getenvint("NUMGAS", 1) ; 

    LOG(info) << " geometry " << geometry << " layers " << layers ;    

    if(strcmp(geometry, "sphere_containing_grid_of_spheres") == 0)
    {
        init_sphere_containing_grid_of_spheres(layers );
    }
    else if(strcmp(geometry, "parade") == 0)
    {
        init_parade();
    }
    else if(SStr::StartsWith(geometry, "clustered_"))
    {
        init_clustered( geometry + strlen("clustered_")); 
    }
    else if(SStr::StartsWith(geometry, "scaled_"))
    {
        init_scaled( geometry, geometry + strlen("scaled_"), outer, layers, numgas ); 
    }
    else if(SStr::StartsWith(geometry, "layered_"))
    {
        init_layered( geometry + strlen("layered_"), outer, layers ); 
    }
    else
    {
        init(geometry); 
    }

    LOG(info) << "]" ; 
}

/**
DemoGeo::init_sphere_containing_grid_of_spheres
---------------------------------------------

A cube of side 1 (halfside 0.5) has diagonal sqrt(3):1.7320508075688772 
that will fit inside a sphere of diameter sqrt(3) (radius sqrt(3)/2 : 0.86602540378443)
Container sphere "extent" needs to be sqrt(3) larger than the grid extent.

**/

void DemoGeo::init_sphere_containing_grid_of_spheres(unsigned layers )
{
    LOG(info) << "layers " << layers  ; 
    maker->makeDemoSolids();  
    unsigned num_solid = foundry->getNumSolid() ; 

    unsigned ias_idx = 0 ; 
    const float4 ce = DemoGrid::AddInstances(foundry, ias_idx, num_solid) ; 

    float big_radius = float(ce.w)*sqrtf(3.f) ;
    LOG(info) << " big_radius " << big_radius ; 


    maker->makeLayered("sphere", 0.7f, layers ); 
    maker->makeLayered("sphere", 1.0f, layers ); 
    maker->makeLayered("sphere", big_radius, 1 ); 
}


void DemoGeo::init_parade()
{
    LOG(info) << "["  ;
 
    maker->makeDemoSolids();  
    unsigned num_solid = foundry->getNumSolid() ; 

    unsigned ias_idx = 0 ; 
    DemoGrid::AddInstances( foundry, ias_idx, num_solid ); 

    LOG(info) << "]"  ; 
}

/**
DemoGeo::init_clustered
--------------------

Aiming to test a GAS containing multiple spread (non-concentric) 
placements of the same type of single node Prim.  
Will need to assign appropriate node transforms and get those applied 
to the bbox at Prim+Node(?) level.

**/
void DemoGeo::init_clustered(const char* name)
{
    float unit = SSys::getenvfloat("CLUSTERUNIT", 200.f ); 
    const char* clusterspec = SSys::getenvvar("CLUSTERSPEC","-1:2:1,-1:2:1,-1:2:1") ; 

    LOG(info) 
        << " name " << name 
        << " clusterspec " << clusterspec 
        << " unit " << unit 
        ; 

    bool inbox = false ; 
    std::array<int,9> cl ; 
    SStr::ParseGridSpec(cl, clusterspec); // string parsed into array of 9 ints 
    CSGSolid* so = maker->makeClustered(name, cl[0],cl[1],cl[2],cl[3],cl[4],cl[5],cl[6],cl[7],cl[8], unit, inbox ); 
    std::cout << "DemoGeo::init_layered" << name << " so.center_extent " << so->center_extent << std::endl ; 

    unsigned gas_idx = 0 ; 
    addInstance(gas_idx);  
}



void DemoGeo::init_scaled(const char* solid_label_base, const char* demo_node_type, float outer, unsigned layers, unsigned num_gas )
{
    for(unsigned gas_idx=0 ; gas_idx < num_gas ; gas_idx++)
    {
         std::string label = CSGSolid::MakeLabel( solid_label_base, gas_idx );   

        CSGSolid* so = maker->makeScaled(label.c_str(), demo_node_type, outer, layers ); 
        LOG(info)
            << " gas_idx " << gas_idx 
            << " label " << label 
            << " demo_node_type " << demo_node_type
            << " so.center_extent " << so->center_extent 
            << " num_gas " << num_gas 
            ; 

        float4 ce = so->center_extent ; 

        float tx = 0.f ; 
        float ty = 3.f*ce.w*gas_idx ;
        float tz = 0.f ;
 
        addInstance(gas_idx, tx, ty, tz);  
    }
}

void DemoGeo::init_layered(const char* name, float outer, unsigned layers)
{
    CSGSolid* so = maker->makeLayered(name, outer, layers ); 
    LOG(info) << " name " << name << " so.center_extent " << so->center_extent ; 

    unsigned gas_idx = 0 ; 
    addInstance(gas_idx);  
}

void DemoGeo::init(const char* name)
{
    CSGSolid* so = maker->make(name) ; 
    LOG(info) << " name " << name << " so.center_extent " << so->center_extent ; 

    unsigned gas_idx = 0 ; 
    addInstance(gas_idx);  
}

void DemoGeo::addInstance(unsigned gas_idx, float tx, float ty, float tz)
{
    unsigned ias_idx = 0 ; 
    unsigned ins_idx = foundry->inst.size() ; // 0-based index within the DemoGrid
    qat4 q  ; 
    q.setIdentity( ins_idx, gas_idx, ias_idx ); 

    q.q3.f.x = tx ; 
    q.q3.f.y = ty ; 
    q.q3.f.z = tz ; 

    foundry->inst.push_back( q ); 
}

std::string DemoGeo::desc() const
{
    std::stringstream ss ; 
    ss << "DemoGeo " ;
    std::string s = ss.str(); 
    return s ; 
}

