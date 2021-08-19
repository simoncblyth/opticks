#include <string>
#include <sstream>
#include <iostream>

#include "SStr.hh"

#include "sutil_vec_math.h"

#include "DemoGrid.h"
#include "AABB.h"

#include "PLOG.hh"
#include "CSGFoundry.h"


float4 DemoGrid::AddInstances( CSGFoundry* foundry_, unsigned ias_idx_, unsigned num_solid_ )  // static 
{
    DemoGrid gr(foundry_, ias_idx_, num_solid_ ); 
    return gr.center_extent(); 
} 

DemoGrid::DemoGrid( CSGFoundry* foundry_, unsigned ias_idx_, unsigned num_solid_ )
    :
    foundry(foundry_),
    ias_idx(ias_idx_),
    num_solid(num_solid_),
    gridscale(SStr::GetEValue<float>("GRIDSCALE", 1.f))
{
    std::string gridspec = SStr::GetEValue<std::string>("GRIDSPEC","-10:11,2,-10:11:2,-10:11,2") ; 
    SStr::ParseGridSpec(grid, gridspec.c_str());      // string parsed into array of 9 ints 
    SStr::GetEVector(solid_modulo, "GRIDMODULO", "0,1" ); 
    SStr::GetEVector(solid_single, "GRIDSINGLE", "2" ); 

    LOG(info) << "GRIDSPEC " << gridspec ; 
    LOG(info) << "GRIDSCALE " << gridscale ; 
    LOG(info) << "GRIDMODULO " << SStr::Present(solid_modulo) ; 
    LOG(info) << "GRIDSINGLE " << SStr::Present(solid_single) ; 

    init();   // add qat4 instances to foundry 
}


const float4 DemoGrid::center_extent() const 
{
    glm::ivec3 imn(0,0,0); 
    glm::ivec3 imx(0,0,0); 
    SStr::GridMinMax(grid, imn, imx); 

    float3 mn = gridscale*make_float3( float(imn.x), float(imn.y), float(imn.z) ) ;
    float3 mx = gridscale*make_float3( float(imx.x), float(imx.y), float(imx.z) ) ;

    // hmm this does not accomodat the bbox of the item, just the grid centers of the items
    AABB bb = { mn, mx }; 
    float4 ce = bb.center_extent(); 

    return ce ; 
}

void DemoGrid::init()
{
    unsigned num_solid_modulo = solid_modulo.size() ; 
    unsigned num_solid_single = solid_single.size() ; 

    LOG(info) 
        << "DemoGrid::init"
        << " num_solid_modulo " << num_solid_modulo
        << " num_solid_single " << num_solid_single
        << " num_solid " << num_solid
        ;

    // check the input solid_idx are valid 
    for(unsigned i=0 ; i < num_solid_modulo ; i++ ) assert(solid_modulo[i] < num_solid) ; 
    for(unsigned i=0 ; i < num_solid_single ; i++ ) assert(solid_single[i] < num_solid) ; 

    for(int i=0 ; i < int(num_solid_single) ; i++)
    {
        unsigned ins_idx = foundry->inst.size() ; // 0-based index within the DemoGrid
        unsigned gas_idx = solid_single[i] ;      // 0-based solid index
        qat4 instance  ; 
        instance.setIdentity( ins_idx, gas_idx, ias_idx ); 
        foundry->inst.push_back( instance ); 
    }

    for(int i=grid[0] ; i < grid[1] ; i+=grid[2] ){
    for(int j=grid[3] ; j < grid[4] ; j+=grid[5] ){
    for(int k=grid[6] ; k < grid[7] ; k+=grid[8] ){

        qat4 instance  ; 
        instance.q3.f.x = float(i)*gridscale ; 
        instance.q3.f.y = float(j)*gridscale ; 
        instance.q3.f.z = float(k)*gridscale ; 
       
        unsigned ins_idx = foundry->inst.size() ;     
        unsigned solid_modulo_idx = ins_idx % num_solid_modulo ; 
        unsigned gas_idx = solid_modulo[solid_modulo_idx] ; 

        instance.setIdentity( ins_idx, gas_idx, ias_idx ); 
        foundry->inst.push_back( instance ); 
    }
    }
    }
}

