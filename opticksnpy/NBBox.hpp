#pragma once

#include <functional>
#include <vector>
#include <glm/fwd.hpp>

struct nmat4triple ; 

#include "OpticksCSG.h"
#include "NQuad.hpp"
#include "NBBoxEnum.hpp"

#include "NPY_API_EXPORT.hh"

struct NPY_API nbbox 
{
    std::function<float(float,float,float)> sdf() const ;
    float operator()(const glm::vec3& q, const nmat4triple* t_=NULL ) const  ;
    float operator()(float x_, float y_, float z_, const nmat4triple* t_=NULL ) const  ;
    float sdf_(const glm::vec3& q, const nmat4triple* t_=NULL ) const  ;
    void scan_sdf( const glm::vec3& o, const glm::vec3& range, const nmat4triple* t=NULL ) const ;


    void dump(const char* msg);
    void include(const nbbox& other );
    void include(const glm::vec3& p);
    static nbbox from_points(const std::vector<glm::vec3>& points);

    const char* desc() const;
    std::string description() const ; 

    static NBBoxContainment_t classify_containment_1( float delta, float epsilon,  NBBoxContainment_t neg, NBBoxContainment_t eps, NBBoxContainment_t pos );
    static std::string containment_mask_string( unsigned mask );
    static const char* containment_name( NBBoxContainment_t cont );
    unsigned classify_containment( const nbbox& container, float epsilon ) const ; // of this bbox against purported container

    // transform returns a transformed copy of the bbox
    nbbox make_transformed( const glm::mat4& t ) const ;
    static void transform_brute(nbbox& tbb, const nbbox& bb, const glm::mat4& t );
    static void transform(nbbox& tbb, const nbbox& bb, const glm::mat4& t );


    static bool HasOverlap(const nbbox& a, const nbbox& b );
    static bool FindOverlap(nbbox& overlap, const nbbox& a, const nbbox& b );
    static void CombineCSG(nbbox& comb, const nbbox& a, const nbbox& b, OpticksCSG_t op, int verbosity );

    bool has_overlap(const nbbox& other);
    bool find_overlap(nbbox& overlap, const nbbox& other);


    void copy_from(const nbbox& src); 


    nvec4 center_extent() const ;
    nvec4 dimension_extent() const ;
    static float extent(const nvec4& dim);

    bool contains( const nvec3& p, float epsilon=1e-4) const ; 
    bool contains( const nbbox& other, float epsilon=1e-4) const ; 

    bool is_empty() const 
    {
        return min.x == 0. && min.y == 0. && min.z == 0. && max.x == 0. && max.y == 0. && max.z == 0.  ; 
    }

    void set_empty()
    {
        min.x = 0. ; 
        min.y = 0. ; 
        min.z = 0. ; 

        max.x = 0. ; 
        max.y = 0. ; 
        max.z = 0. ; 
    } 

    nvec3 side() const 
    {
        return max - min ; 
    }

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
    bool  invert ; 
};


inline NPY_API void nbbox::copy_from(const nbbox& src)
{
    min = src.min ; 
    max = src.max ; 
    invert = src.invert ; 
}

inline NPY_API bool operator == (const nbbox& a , const nbbox& b )
{
   return a.min == b.min && a.max == b.max ; 
}

inline NPY_API nbbox make_bbox()
{
    nbbox bb ; 
    bb.set_empty();
    bb.invert = false ; 
    return bb ; 
}

// "ctor" assuming rotational symmetry around z axis
inline NPY_API nbbox make_bbox_zsymmetric(float zmin, float zmax, float ymin, float ymax)
{
    nbbox bb = make_bbox();
    bb.min = make_nvec3( ymin, ymin, zmin ) ;
    bb.max = make_nvec3( ymax, ymax, zmax ) ;
    bb.invert = false ; 

    return bb ;
}

inline NPY_API nbbox make_bbox(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax)
{
    nbbox bb = make_bbox();

    assert( xmax >= xmin );
    assert( ymax >= ymin );
    assert( zmax >= zmin );

    bb.min = {xmin,ymin,zmin} ;
    bb.max = {xmax,ymax,zmax} ;

    return bb ; 
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

