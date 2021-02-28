#include <string>
#include <sstream>
#include <iostream>
#include "NP.hh"
#include "Sys.h"
#include "Util.h"
#include "Grid.h"
#include <glm/gtx/transform.hpp>


/**
Geo::makeGrid
---------------

shape_modulo
    vector of gas_idx which are modulo cycled in a 3d grid array  

shape_single
    vector of gas_idx which are singly included into the IAS 
    with an identity transform

Create vector of transfoms and creat IAS from that.
Currently a 3D grid of translate transforms with all available GAS repeated modulo

**/


Grid::Grid( unsigned num_shape_ )
    :
    num_shape(num_shape_)
{
    std::string gridspec = Util::GetEValue<std::string>("GRIDSPEC","-10:11,2,-10:11:2,-10:11,2") ; 
    Util::ParseGridSpec(grid, gridspec.c_str());     
    Util::GetEVector(shape_modulo, "MODULO", "0,1" ); 
    Util::GetEVector(shape_single, "SINGLE", "2" ); 

    std::cout << "GRIDSPEC " << gridspec << std::endl ; 
    std::cout << "MODULO " << Util::Present(shape_modulo) << std::endl ; 
    std::cout << "SINGLE " << Util::Present(shape_single) << std::endl ; 

    init(); 
}

int Grid::extent() const 
{
    int mn(0); 
    int mx(0); 
    Util::GridMinMax(grid, mn, mx); 
    int grid_extent_ = std::max( std::abs(mn), std::abs(mx) );  // half side
    return grid_extent_ ; 
}

std::string Grid::desc() const 
{
    std::stringstream ss ; 
    ss << "Grid extent " << extent() << " num_tr " << trs.size() ; 
    std::string s = ss.str(); 
    return s; 
}

void Grid::init()
{
    unsigned num_shape_modulo = shape_modulo.size() ; 
    unsigned num_shape_single = shape_single.size() ; 

    // check the input shape_idx are valid 
    for(unsigned i=0 ; i < num_shape_modulo ; i++ ) assert(shape_modulo[i] < num_shape) ; 
    for(unsigned i=0 ; i < num_shape_single ; i++ ) assert(shape_single[i] < num_shape) ; 

    std::cout 
        << "Grid::init"
        << " num_shape_modulo " << num_shape_modulo
        << " num_shape_single " << num_shape_single
        << " num_shape " << num_shape
        << std::endl
        ;

    for(int i=0 ; i < int(num_shape_single) ; i++)
    {
        unsigned transform_idx = trs.size() ;  // 0-based index within the Grid
        unsigned shape_idx = shape_single[i] ; 

        glm::mat4 tr(1.f) ;  // identity transform for the large sphere 
        tr[0][3] = Sys::unsigned_as_float(transform_idx); 
        tr[1][3] = Sys::unsigned_as_float(shape_idx) ;
        tr[2][3] = Sys::unsigned_as_float(0) ;   
        tr[3][3] = Sys::unsigned_as_float(0) ;   

        trs.push_back(tr); 
    }

    for(int i=grid[0] ; i < grid[1] ; i+=grid[2] ){
    for(int j=grid[3] ; j < grid[4] ; j+=grid[5] ){
    for(int k=grid[6] ; k < grid[7] ; k+=grid[8] ){

        glm::vec3 tlat(i*1.f,j*1.f,k*1.f) ;  // grid translation 
        glm::mat4 tr(1.f) ;
        tr = glm::translate(tr, tlat );

        unsigned transform_idx = trs.size();   // 0-based instance index within the IAS
        unsigned shape_modulo_idx = transform_idx % num_shape_modulo ; 
        unsigned shape_idx = shape_modulo[shape_modulo_idx] ; 

        tr[0][3] = Sys::unsigned_as_float(transform_idx); 
        tr[1][3] = Sys::unsigned_as_float(shape_idx) ;
        tr[2][3] = Sys::unsigned_as_float(0) ;   
        tr[3][3] = Sys::unsigned_as_float(0) ;   

        trs.push_back(tr); 
    }
    }
    }
}


void Grid::write(const char* base, const char* rel, unsigned idx ) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel << "/" << idx << "/" ; 
    std::string dir = ss.str();   

    std::cout 
        << "Grid::write "
        << " trs.size " << trs.size()
        << " dir " << dir
        << std::endl 
        ;

    NP::Write(dir.c_str(), "grid.npy", (float*)trs.data(),  trs.size(), 4, 4 ); 
}


