/**
CSGIntersectComparisonTest.cc
===============================

::

    RAYORI=90,0,0 RAYDIR=1,0,0 TMIN=9 CSGIntersectComparisonTest

**/


#include "OPTICKS_LOG.hh"
#include <cmath>
#include <vector>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "ssincos.h"

#include "SSys.hh"
#include "SRng.hh"
#include "SPath.hh"
#include "NP.hh"

#define DEBUG 1 
#include "csg_intersect_leaf.h"

#include "CSGNode.h"

enum { X, Y, Z } ; 


struct CSGIntersectComparisonTest
{
    static constexpr const char* FOLD = "$TMP/CSGIntersectComparisonTest" ; 

    const CSGNode* a ; 
    const CSGNode* b ; 
    float3 a_mn ; 
    float3 a_mx ; 
    float3 b_mn ; 
    float3 b_mx ; 
    const float* mn ; 
    const float* mx ; 

    float4 a_isect ; 
    float4 b_isect ; 
    bool a_valid_isect ; 
    bool b_valid_isect ; 
    int  status ; 
    float epsilon ; 
    bool expected ; 

    float4 zero4 ; 
    float3 zero3 ; 
    float3 a_pos ; 
    float3 b_pos ; 

    float t_min ; 
    float3 ray_origin ; 
    float3 ray_direction ; 
    float* oo ; 
    float* vv ; 

    unsigned seed ; 
    SRng<double> rng ; 
    std::vector<quad4> a_simtrace ;  
    std::vector<quad4> b_simtrace ;  


    CSGIntersectComparisonTest( const CSGNode* a, const CSGNode* b ); 
    static void Zero(float4& v); 
    static void Zero(float3& v); 
    void zero(); 

    void intersect(); 
    static void FormSimtrace( quad4& st, const float4& isect, const float3& pos, float t_min, const float3& ori, const float3& dir ); 
    void save(); 

    bool is_expected() const ; 

    float dpos() const ; 
    float ddis() const ; 
    float dnrm() const ; 
    float dmax() const ; 
    std::string descDiff() const ; 

    void scan(int axis); 

    static unsigned OtherAxis(unsigned h, unsigned v); 
    void random2D(unsigned h, unsigned v); 

    std::string descGeom() const ; 
    std::string descRay() const ; 
    std::string descA() const ; 
    std::string descB() const ; 
    std::string descFull() const ; 
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
    a_mn(a->mn()),
    a_mx(a->mx()),
    b_mn(b->mn()),
    b_mx(b->mx()),
    mn((const float*)&a_mn),
    mx((const float*)&a_mx),
    a_isect( make_float4(0.f, 0.f, 0.f, 0.f) ),
    b_isect( make_float4(0.f, 0.f, 0.f, 0.f) ),
    a_valid_isect(false),
    b_valid_isect(false),
    status(-1),
    epsilon(SSys::getenvfloat("EPSILON", 1e-5)),
    expected(true),
    zero4( make_float4( 0.f, 0.f, 0.f, 0.f )), 
    zero3( make_float3( 0.f, 0.f, 0.f )), 
    a_pos( make_float3( 0.f, 0.f, 0.f )),
    b_pos( make_float3( 0.f, 0.f, 0.f )),
    t_min(SSys::getenvint("TMIN",0.f)),
    ray_origin(   SSys::getenvfloat3("RAYORI","0,0,0")), 
    ray_direction(SSys::getenvfloat3norm("RAYDIR","0,0,1")),
    oo((float*)&ray_origin),
    vv((float*)&ray_direction),
    seed(SSys::getenvunsigned("SEED", 1u)),
    rng(seed)
{
    std::cout << descGeom() ; 

    assert( a_mn.x == b_mn.x ); 
    assert( a_mn.y == b_mn.y ); 
    assert( a_mn.z == b_mn.z );

    assert( a_mx.x == b_mx.x ); 
    assert( a_mx.y == b_mx.y ); 
    assert( a_mx.z == b_mx.z );
}

void CSGIntersectComparisonTest::Zero(float4& v)
{
    v.x = 0.f ; 
    v.y = 0.f ; 
    v.z = 0.f ; 
    v.w = 0.f ; 
}
void CSGIntersectComparisonTest::Zero(float3& v)
{
    v.x = 0.f ; 
    v.y = 0.f ; 
    v.z = 0.f ; 
}
void CSGIntersectComparisonTest::zero()
{
    Zero(a_isect); 
    Zero(b_isect);
    Zero(a_pos);  
    Zero(b_pos);  
}

void CSGIntersectComparisonTest::intersect()
{
    zero(); 
    a_valid_isect = intersect_leaf( a_isect, a , nullptr, nullptr, t_min, ray_origin, ray_direction ); 
    b_valid_isect = intersect_leaf( b_isect, b , nullptr, nullptr, t_min, ray_origin, ray_direction ); 
    status = ( int(a_valid_isect) << 1 ) | int(b_valid_isect) ;   // 0: MISS MISS, 3: HIT HIT, 1/2 mixed   
    a_pos = a_valid_isect ? ray_origin + a_isect.w*ray_direction : zero3 ;  
    b_pos = b_valid_isect ? ray_origin + b_isect.w*ray_direction : zero3 ;  

    expected = is_expected();   // needs a_pos b_pos
    if(expected == false) std::cout << desc() << std::endl; 

    quad4 sta ; 
    FormSimtrace(sta, a_isect, a_pos, t_min, ray_origin, ray_direction ); 
    a_simtrace.push_back(sta); 

    quad4 stb ; 
    FormSimtrace(stb, b_isect, b_pos, t_min, ray_origin, ray_direction ); 
    b_simtrace.push_back(stb); 
}

void CSGIntersectComparisonTest::save()
{
    const char* fold = SPath::Resolve(FOLD, DIRPATH); 
    LOG(info) << fold ; 
    NP::Write(fold, "a_simtrace.npy",  (float*)a_simtrace.data(), a_simtrace.size(), 4, 4 ); 
    NP::Write(fold, "b_simtrace.npy",  (float*)b_simtrace.data(), b_simtrace.size(), 4, 4 ); 
}

void CSGIntersectComparisonTest::FormSimtrace( quad4& st, const float4& isect, const float3& pos, float t_min, const float3& ori, const float3& dir )
{
    st.q0.f = isect ; 

    st.q1.f.x = pos.x ; 
    st.q1.f.y = pos.y ; 
    st.q1.f.z = pos.z ;
    st.q1.f.w = t_min ;

    st.q2.f.x = ori.x ; 
    st.q2.f.y = ori.y ; 
    st.q2.f.z = ori.z ;
    st.q2.f.w = 0.f ;

    st.q3.f.x = dir.x ; 
    st.q3.f.y = dir.y ; 
    st.q3.f.z = dir.z ;
    st.q3.f.w = 0.f ;
}


bool CSGIntersectComparisonTest::is_expected() const
{
    float dm = dmax(); 
    bool expected_status = status == 0 || status == 3 ; 
    bool expected_diff   = status == 3 ? fabs(dm) < epsilon : true  ; 
    return expected_status && expected_diff ; 
}

float CSGIntersectComparisonTest::dpos() const
{
    float3 d_pos = a_pos - b_pos ; 
    return fmaxf(d_pos) ;  
}
float CSGIntersectComparisonTest::ddis() const
{
    return a_isect.w - b_isect.w  ;  
}
float CSGIntersectComparisonTest::dnrm() const
{
    float4 d_isect = a_isect - b_isect ; 
    float3* nrm = (float3*)&d_isect ; 
    return fmaxf(*nrm)  ;  
}
float CSGIntersectComparisonTest::dmax() const
{
    float3 dm = make_float3( dpos(), ddis(), dnrm() ); 
    return fmaxf(dm); 
}
std::string CSGIntersectComparisonTest::descDiff() const
{
    float dp = dpos(); 
    float dt = ddis(); 
    float dn = dnrm(); 
    float dm = dmax(); 
    std::stringstream ss ; 
    ss << " status " << status ; 
    ss << " dpos " << std::scientific << dp ;  
    ss << " ddis " << std::scientific << dt ;  
    ss << " dnrm " << std::scientific << dn ;  
    ss << " dmax " << std::scientific << dm ;  
    std::string s = ss.str();
    return s ; 
}



std::string CSGIntersectComparisonTest::descGeom() const 
{
    std::stringstream ss ; 
    ss << " A " << a->desc() << std::endl ; 
    ss << " B " << b->desc() << std::endl ; 
    ss << " a_mn " << Desc(a_mn) << std::endl ; 
    ss << " b_mn " << Desc(b_mn) << std::endl ; 
    ss << " a_mx " << Desc(a_mx) << std::endl ; 
    ss << " b_mx " << Desc(b_mx) << std::endl ; 
    std::string s = ss.str();
    return s ; 
}
std::string CSGIntersectComparisonTest::descRay() const 
{
    return DescRay(ray_origin, ray_direction, t_min) ; 
}
std::string CSGIntersectComparisonTest::descA() const 
{
    std::stringstream ss ; 
    ss << " A " << ( a_valid_isect ? DescIsect(a_isect, a_pos) : "MISS" ) ;     
    std::string s = ss.str();
    return s ; 
}
std::string CSGIntersectComparisonTest::descB() const 
{
    std::stringstream ss ; 
    ss << " B " << ( b_valid_isect ? DescIsect(b_isect, b_pos) : "MISS" ) ;     
    std::string s = ss.str();
    return s ; 
}
std::string CSGIntersectComparisonTest::descFull() const
{
    std::stringstream ss ; 
    ss << descRay() << descA() << descB() << descDiff() ; 
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::desc() const
{
    std::stringstream ss ; 
    ss << descRay() << descA() << descDiff() ; 
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
void CSGIntersectComparisonTest::scan(int axis)
{
    float v_mn = mn[axis]*1.2f ; 
    float v_mx = mx[axis]*1.2f ; 

    float v_st = (v_mx - v_mn)/100.f ;   

    for(float v=v_mn ; v <= v_mx ; v+= v_st )
    {
        oo[axis] = v ; 
        intersect(); 
    }
}

unsigned CSGIntersectComparisonTest::OtherAxis(unsigned h, unsigned v) // static
{
    // identify the other axis 
    unsigned o = 0u ; 
    if( h == 0u && v == 1u ) o = 2u ; 
    if( h == 0u && v == 2u ) o = 1u ; 
    if( h == 1u && v == 2u ) o = 0u ; 
    if( v == 0u && h == 1u ) o = 2u ; 
    if( v == 0u && h == 2u ) o = 1u ; 
    if( v == 1u && h == 2u ) o = 0u ; 
    return o ; 
}

void CSGIntersectComparisonTest::random2D(unsigned h, unsigned v)
{
    assert( h != v ); 
    assert( h < 3 ); 
    assert( v < 3 ); 
    unsigned o = OtherAxis(h,v); 
    unsigned n = SSys::getenvunsigned("NUM", 1000) ; 

    float margin = 1.2f ; 

    float h0 = mn[h]*margin ; 
    float h1 = mx[h]*margin ; 

    float v0 = mn[v]*margin ; 
    float v1 = mx[v]*margin ; 

    float o0 = mn[o]*margin ; 
    float o1 = mx[o]*margin ; 
    float ao = (o0 + o1)/2.f  ; 


    double phi, sinPhi, cosPhi ; 


    for(unsigned i=0 ; i < n ; i++ )
    {
        double u0 = rng() ; 
        double u1 = rng() ; 
        double u2 = rng() ; 

        double ah = h0 + u0*(h1 - h0 ); 
        double av = v0 + u1*(v1 - v0 ); 

        oo[h] = ah ; 
        oo[v] = av ; 
        oo[o] = ao ; 
        
        phi = 2.*M_PIf*u2 ;     // azimuthal 0->2pi 
        ssincos(phi,sinPhi,cosPhi);  

        vv[h] = cosPhi ; 
        vv[v] = sinPhi ; 
        vv[o] = 0.f ; 
   
        intersect(); 
    }
}



void test_one_expected_miss( CSGIntersectComparisonTest& t )
{
    t.ray_direction.x = 0.993f ; 
    t.ray_direction.y = 0.f ; 
    t.ray_direction.z = 0.119f ;   // HMM: normalize ?

    t.ray_origin.x =  110.f ; 
    t.ray_origin.y =    0.f ; 
    t.ray_origin.z =   40.f ; 

    t.t_min = 0.f ; 

    t.intersect(); 
}
void test_one_expected_hit( CSGIntersectComparisonTest& t )
{
    t.ray_direction = normalize(make_float3( 1.f, 1.f, 1.f) ); 
    t.ray_origin    = make_float3( 0.f, 0.f, 0.f ); 
    t.intersect(); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    float hz = 0.1f ; 
    //float hz = 50.f ; 

    float radius = 100.f ; 
    float z2 = hz ; 
    float z1 = -hz ; 

    CSGNode a = CSGNode::Cylinder( 0.f, 0.f, radius, z1, z2 ) ; 
    CSGNode b = CSGNode::AltCylinder( radius, z1, z2 ) ; 

    CSGIntersectComparisonTest t(&a, &b); 

    //test_one_expected_miss(t); 
    //test_one_expected_hit(t); 

    //t.scan(X) ; 
    t.random2D(X,Z) ; 

    t.save(); 

    return 0 ; 
}



