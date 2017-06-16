#pragma once

#include <glm/fwd.hpp>

#include "OpticksCSG.h"
#include "NQuad.hpp"
#include "NPY_API_EXPORT.hh"


struct NPY_API nbbox 
{
    void dump(const char* msg);
    void include(const nbbox& other );
    const char* desc() const;
    std::string description() const ; 


    // transform returns a transformed copy of the bbox
    nbbox transform( const glm::mat4& t );
    static void transform_brute(nbbox& tbb, const nbbox& bb, const glm::mat4& t );
    static void transform(nbbox& tbb, const nbbox& bb, const glm::mat4& t );


    static bool HasOverlap(const nbbox& a, const nbbox& b );
    static bool FindOverlap(nbbox& overlap, const nbbox& a, const nbbox& b );
    static void CombineCSG(nbbox& comb, const nbbox& a, const nbbox& b, OpticksCSG_t op, int verbosity );

    bool has_overlap(const nbbox& other);
    bool find_overlap(nbbox& overlap, const nbbox& other);





    nvec4 center_extent() const ;
    nvec4 dimension_extent() const ;
    static float extent(const nvec4& dim);

    bool contains( const nvec3& p, float epsilon=1e-4) const ; 
    bool contains( const nbbox& other, float epsilon=1e-4) const ; 


    void expand(float delta)
    {
        min -= delta ; 
        max += delta ; 
        side = max - min ; 
    } 

    void scale(float factor)
    {
        min *= factor ; 
        max *= factor ; 
        side = max - min ; 
    } 

    nvec3 min ; 
    nvec3 max ; 
    nvec3 side ; 
    bool  invert ; 
    bool  empty ; 
};


inline NPY_API bool operator == (const nbbox& a , const nbbox& b )
{
   return a.min == b.min && a.max == b.max && a.side == b.side ;  
}




// "ctor" assuming rotational symmetry around z axis
inline NPY_API nbbox make_bbox(float zmin, float zmax, float ymin, float ymax)
{
    nbbox bb ; 

    bb.min = make_nvec3( ymin, ymin, zmin ) ;
    bb.max = make_nvec3( ymax, ymax, zmax ) ;
    bb.side = bb.max - bb.min ; 
    bb.invert = false ; 
    bb.empty = false ; 

    return bb ;
}


inline NPY_API nbbox make_bbox()
{
    return make_bbox(0,0,0,0) ;
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


