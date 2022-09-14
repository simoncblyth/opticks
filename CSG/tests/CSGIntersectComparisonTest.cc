/**
CSGIntersectComparisonTest.cc
===============================

::

    RAYORI=90,0,0 RAYDIR=1,0,0 TMIN=9 CSGIntersectComparisonTest

**/


#include "OPTICKS_LOG.hh"
#include <cmath>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#include "SSys.hh"

#define DEBUG 1 
#include "csg_intersect_leaf.h"

#include "CSGNode.h"

struct CSGIntersectComparisonTest
{
    const CSGNode* a ; 
    const CSGNode* b ; 
    float4 a_isect ; 
    float4 b_isect ; 
    bool a_valid_isect ; 
    bool b_valid_isect ; 

    float t_min ; 
    float3 ray_origin ; 
    float3 ray_direction ; 

    CSGIntersectComparisonTest( const CSGNode* a, const CSGNode* b ); 
    void intersect(); 
    void xscan(); 

    std::string descGeom() const ; 
    std::string descRay() const ; 
    std::string descA() const ; 
    std::string descB() const ; 
    std::string desc() const ; 

    static std::string DescRay(const float3& o, const float3& v, const float t_min ); 
    static std::string DescIsect(const float4& isect, const float3& pos ); 
    static std::string Desc(const float& v); 
    static std::string Desc(const float3& v); 
    static std::string Desc(const float4& v); 
}; 

CSGIntersectComparisonTest::CSGIntersectComparisonTest( const CSGNode* a_, const CSGNode* b_ ) 
    : 
    a(a_), 
    b(b_),
    a_isect( make_float4(0.f, 0.f, 0.f, 0.f) ),
    b_isect( make_float4(0.f, 0.f, 0.f, 0.f) ),
    a_valid_isect(false),
    b_valid_isect(false),
    t_min(SSys::getenvint("TMIN",0.f)),
    ray_origin(   SSys::getenvfloat3("RAYORI","0,0,0")), 
    ray_direction(SSys::getenvfloat3("RAYDIR","0,0,1")) 
{
    std::cout << descGeom() ; 
}

void CSGIntersectComparisonTest::intersect()
{
    a_valid_isect = intersect_leaf( a_isect, a , nullptr, nullptr, t_min, ray_origin, ray_direction ); 
    b_valid_isect = intersect_leaf( b_isect, b , nullptr, nullptr, t_min, ray_origin, ray_direction ); 

    std::cout << desc() << std::endl; 
}


std::string CSGIntersectComparisonTest::descGeom() const 
{
    std::stringstream ss ; 
    ss << " A " << a->desc() << std::endl ; 
    ss << " B " << b->desc() << std::endl ; 
    std::string s = ss.str();
    return s ; 
}
std::string CSGIntersectComparisonTest::descRay() const 
{
    return DescRay(ray_origin, ray_direction, t_min) ; 
}

std::string CSGIntersectComparisonTest::descA() const 
{
    float3 a_pos = ray_origin + a_isect.w*ray_direction ;  
    std::stringstream ss ; 
    ss << " A " << ( a_valid_isect ? DescIsect(a_isect, a_pos) : "MISS" ) ;     
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::descB() const 
{
    float3 b_pos = ray_origin + b_isect.w*ray_direction ;  
    std::stringstream ss ; 
    ss << " B " << ( b_valid_isect ? DescIsect(b_isect, b_pos) : "MISS" ) ;     
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::desc() const
{
    std::stringstream ss ; 
    ss << descRay() << descA() << descB() ; 
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::Desc(const float& v) // static 
{
    std::stringstream ss ; 
    ss <<  " " << std::fixed << std::setw(7) << std::setprecision(2) << v ; 
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::DescRay(const float3& o, const float3& v, const float t_min ) // static 
{
    std::stringstream ss ; 
    ss 
       << "("
       << Desc(o.x) 
       << Desc(o.y) 
       << Desc(o.z)
       << ";"
       << Desc(v.x) 
       << Desc(v.y) 
       << Desc(v.z)
       << ";"
       << Desc(t_min)
       << ")"
       ;
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::DescIsect(const float4& isect, const float3& pos ) // static 
{
    std::stringstream ss ; 
    ss 
       << "("
       << Desc(isect.x) 
       << Desc(isect.y) 
       << Desc(isect.z)
       << Desc(isect.w)
       << ";"
       << Desc(pos.x) 
       << Desc(pos.y) 
       << Desc(pos.z)
       << ")"
       ;
    std::string s = ss.str();
    return s ; 
}

 


std::string CSGIntersectComparisonTest::Desc(const float3& v) // static 
{
    std::stringstream ss ; 
    ss 
       << "("
       << Desc(v.x) 
       << Desc(v.y) 
       << Desc(v.z) 
       << ")"
       ;
 
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::Desc(const float4& v) // static 
{
    std::stringstream ss ; 
    ss 
       << "("
       << Desc(v.x) 
       << Desc(v.y) 
       << Desc(v.z) 
       << Desc(v.w) 
       << ")"
       ;
    std::string s = ss.str();
    return s ; 
}

void CSGIntersectComparisonTest::xscan()
{
    // TODO: avoid hardcoding sizes by using the bbox of the CSGNode 
    // TODO: axis generalization 

    for(float x=-150.f ; x <= 150.f ; x+= 10.f )
    {
        ray_origin.x = x ; 
        intersect(); 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    float radius = 100.f ; 
    float z2 = 50.f ; 
    float z1 = -50.f ; 

    CSGNode a = CSGNode::Cylinder( 0.f, 0.f, radius, z1, z2 ) ; 
    CSGNode b = CSGNode::AltCylinder( radius, z1, z2 ) ; 

    CSGIntersectComparisonTest t(&a, &b); 

    t.ray_direction.x = 0.993f ; 
    t.ray_direction.y = 0.f ; 
    t.ray_direction.z = 0.119f ; 

    t.ray_origin.x =  110.f ; 
    t.ray_origin.y =    0.f ; 
    t.ray_origin.z =   40.f ; 

    t.t_min = 0.f ; 

    t.intersect(); 

    return 0 ; 
}



