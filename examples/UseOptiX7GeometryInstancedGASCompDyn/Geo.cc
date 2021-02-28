#include <iostream>
#include <iomanip>
#include <cstring>

#include "Sys.h"
#include "Util.h"
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "NP.hh"
#include "Geo.h"
#include "Shape.h"
#include "Grid.h"

Geo* Geo::fGeo = NULL ; 

Geo::Geo()
    :
    top("i0")
{
    fGeo = this ; 
    init();
}

void Geo::init()
{
    float tminf(0.1) ; 
    float tmaxf(10000.f) ; 

    std::string geometry = Util::GetEValue<std::string>("GEOMETRY", "sphere_containing_grid_of_spheres" ); 

    unsigned layers = Util::GetEValue<unsigned>("LAYERS", 1) ; 

    if(strcmp(geometry.c_str(), "sphere_containing_grid_of_spheres") == 0)
    {
        init_sphere_containing_grid_of_spheres(tminf, tmaxf, layers );
    }
    else if(strcmp(geometry.c_str(), "sphere") == 0 )
    {
        init_sphere(tminf, tmaxf, layers);
    }
    else
    {
        assert(0); 
    }

    float top_extent = getTopExtent(); 
    tmin = top_extent*tminf ; 
    tmax = top_extent*tmaxf ; 
    std::cout 
        << "Geo::init" 
        << " top_extent " << top_extent  
        << " tminf " << tminf 
        << " tmin " << tmin 
        << " tmaxf " << tmaxf 
        << " tmax " << tmax 
        << std::endl 
        ; 

    float e_tminf = Util::GetEValue<float>("TMIN", -1.0) ; 
    if(e_tminf > 0.f )
    {
        tmin = top_extent*e_tminf ; 
        std::cout << "Geo::init e_tminf TMIN " << e_tminf << " override tmin " << tmin << std::endl ; 
    }
    
    float e_tmaxf = Util::GetEValue<float>("TMAX", -1.0) ; 
    if(e_tmaxf > 0.f )
    {
        tmax = top_extent*e_tmaxf ; 
        std::cout << "Geo::init e_tmaxf TMAX " << e_tmaxf << " override tmax " << tmax << std::endl ; 
    }
}

/**
Geo::init_sphere_containing_grid_of_spheres
---------------------------------------------

A cube of side 1 (halfside 0.5) has diagonal sqrt(3):1.7320508075688772 
that will fit inside a sphere of diameter sqrt(3) (radius sqrt(3)/2 : 0.86602540378443)
Container sphere "extent" needs to be sqrt(3) larger than the grid extent.

**/

void Geo::init_sphere_containing_grid_of_spheres(float& tminf, float& tmaxf, unsigned layers )
{
    std::cout << "Geo::init_sphere_containing_grid_of_spheres : layers " << layers << std::endl ; 

    Grid* grid = new Grid(3) ; 
    addGrid(grid); 

    float big_radius = float(grid->extent())*sqrtf(3.f) ;
    std::cout << " big_radius " << big_radius << std::endl ; 

    addShape("S", 0.7f, layers); 
    addShape("S", 1.0f, layers); 
    addShape("S", big_radius, 1u); 

    top = strdup("i0") ; 

    setTopExtent(big_radius); 

    tminf = 0.75f ; 
    tmaxf = 10000.f ; 
}


void Geo::init_sphere(float& tminf, float& tmaxf, unsigned layers)
{
    std::cout << "Geo::init_sphere" << std::endl ; 
    addShape("S", 100.f, layers); 
    setTopExtent(100.f); 
    top = strdup("g0") ; 

    tminf = 1.60f ;   //  hmm depends on viewpoint, aiming to cut into the sphere with the tmin
    tmaxf = 10000.f ; 
}

std::string Geo::desc() const
{
    std::stringstream ss ; 
    ss << "Geo shapes: " << shapes.size() << " grids:" << grids.size() ; 
    std::string s = ss.str(); 
    return s ; 
}

Geo* Geo::Get()
{
    return fGeo ; 
}
void Geo::setTopExtent(float top_extent_)
{
    top_extent = top_extent_ ; 
}
float Geo::getTopExtent() const 
{
    return top_extent ; 
}
void Geo::addShape(const char* typs, float outer_extent, unsigned layers)
{
    std::vector<float> extents ;
    for(unsigned i=0 ; i < layers ; i++) extents.push_back(outer_extent/float(i+1));  
    addShape(typs, extents); 
}
void Geo::addShape(const char* typs, const std::vector<float>& extents)
{
    Shape* sh = new Shape(typs, extents) ; 
    shapes.push_back(sh); 
}
void Geo::addGrid(const Grid* grid)
{
    grids.push_back(grid); 
}

unsigned Geo::getNumShape() const 
{
    return shapes.size() ; 
}
unsigned Geo::getNumGrid() const 
{
    return grids.size() ; 
}

const Shape* Geo::getShape(int shape_idx_) const
{
    unsigned shape_idx = shape_idx_ < 0 ? shapes.size() + shape_idx_ : shape_idx_ ;   // -ve counts from end
    assert( shape_idx < shapes.size() ); 
    return shapes[shape_idx] ; 
}
const Grid* Geo::getGrid(int grid_idx_) const
{
    unsigned grid_idx = grid_idx_ < 0 ? grids.size() + grid_idx_ : grid_idx_ ;  
    assert( grid_idx < grids.size() ); 
    return grids[grid_idx] ; 
}

void Geo::write(const char* dir) const 
{
    std::cout << "Geo::write " << dir << std::endl ;  
    unsigned num_shape = getNumShape(); 
    for(unsigned i=0 ; i < num_shape ; i++) 
    {
        const Shape* sh = getShape(i); 
        sh->write(dir,"shape", i ); 
    } 
    unsigned num_grid = getNumGrid(); 
    for(unsigned i=0 ; i < num_grid ; i++) 
    {
        const Grid* gr = getGrid(i); 
        gr->write(dir,"grid", i); 
    }
}
