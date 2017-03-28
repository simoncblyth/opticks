#pragma once

#include "NQuad.hpp"
#include "NPY_API_EXPORT.hh"


struct NPY_API nbbox {

    // NO CTOR

    void dump(const char* msg);
    void include(const nbbox& other );
    const char* desc() const;

    nvec4 center_extent() const ;
    nvec4 dimension_extent() const ;
    static float extent(const nvec4& dim);

    bool contains( const nvec3& p) const ; 
    bool contains( const nbbox& other) const ; 


    void expand(float delta)
    {
        min -= delta ; 
        max += delta ; 
    } 

    void scale(float factor)
    {
        min *= factor ; 
        max *= factor ; 
    } 

    nvec3 min ; 
    nvec3 max ; 
    nvec3 side ; 
};


// "ctor" assuming rotational symmetry around z axis
inline NPY_API nbbox make_nbbox(float zmin, float zmax, float ymin, float ymax)
{
    nbbox bb ; 
    bb.min = make_nvec3( ymin, ymin, zmin ) ;
    bb.max = make_nvec3( ymax, ymax, zmax ) ;

    return bb ;
}


inline NPY_API nbbox make_nbbox()
{
    return make_nbbox(0,0,0,0) ;
}


inline NPY_API bool nbbox::contains(const nvec3& p) const 
{
    return p.x >= min.x && p.x <= max.x &&
           p.y >= min.y && p.y <= max.y &&
           p.z >= min.z && p.z <= max.z ;
} 

inline NPY_API bool nbbox::contains(const nbbox& other) const
{
    return contains( other.min ) && contains(other.max ) ;
} 







inline NPY_API float nbbox::extent(const nvec4& dim) 
{
    float _extent(0.f) ;
    _extent = nmaxf( dim.x , _extent );
    _extent = nmaxf( dim.y , _extent );
    _extent = nmaxf( dim.z , _extent );
    _extent = _extent / 2.0f ;    
    return _extent ; 
}

inline NPY_API nvec4 nbbox::dimension_extent() const
{
    nvec4 de ; 
    de.x = max.x - min.x ; 
    de.y = max.y - min.y ; 
    de.z = max.z - min.z ; 
    de.w = extent(de) ; 
    return de ; 
}

inline NPY_API nvec4 nbbox::center_extent() const 
{
    nvec4 ce ; 
    ce.x = (min.x + max.x)/2.f ;
    ce.y = (min.y + max.y)/2.f ;
    ce.z = (min.z + max.z)/2.f ;
    nvec4 de = dimension_extent();
    ce.w = de.w ;  
    return ce ; 
}


