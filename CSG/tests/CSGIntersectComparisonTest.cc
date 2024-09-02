/**
CSGIntersectComparisonTest.cc
===============================

Compare intersects onto two implementations of a CSGNode 

::

    METHOD=intersect RAYORI=90,0,0 RAYDIR=1,0,0 TMIN=9 CSGIntersectComparisonTest

**/


#include "OPTICKS_LOG.hh"
#include <cmath>
#include <vector>
#include <map>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "ssincos.h"

#include "SSys.hh"
#include "SRng.hh"
#include "SPath.hh"
#include "NP.hh"

//#define DEBUG 1 
#include "csg_intersect_leaf.h"

#include "CSGNode.h"

enum { X, Y, Z } ; 


struct CSGIntersectComparisonTest
{
    static constexpr const char* FOLD = "$TMP/CSGIntersectComparisonTest" ; 

    const char* METHOD ; 
    bool FULL ; 
    const char* RERUN ; 
    NP* rerun ; 
    const quad4* rr ; 

    const char* AXES ; 
    unsigned h ; 
    unsigned v ; 
    unsigned o ; 

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
    float  a_sd ; 
    float  b_sd ; 
    unsigned sd_winner ; 
    std::map<unsigned, unsigned> sd_winner_stats ; 

    float t_min ; 
    float3 ray_origin ; 
    float3 ray_direction ; 
    float* oo ; 
    float* vv ; 

    unsigned seed ; 
    SRng<double> rng ; 
    float margin ; 

    float x0 ; 
    float x1 ; 
    float y0 ; 
    float y1 ; 
    float z0 ; 
    float z1 ; 

    float h0 ; 
    float h1 ; 
    float v0 ; 
    float v1 ; 
    float o0 ; 
    float o1 ; 

    unsigned num ; 

    std::vector<quad4> a_simtrace ;  
    std::vector<quad4> b_simtrace ;  


    CSGIntersectComparisonTest( const CSGNode* a, const CSGNode* b ); 
    void init(); 
    std::string descStats() const ; 

    static void Zero(float4& v); 
    static void Zero(float3& v); 
    void zero(); 

    void intersect(); 
    unsigned smaller_sd() const ; 

    static void LoadSimtrace( const quad4& st, float& t_min, float3& ori, float3& dir ); 
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

    void random2D(); 
    void random2D_ray_origin();
    void random2D_ray_direction(); 

    void random2D_fixed_origin(); 
 
    void random3D(); 
    void random3D_ray_origin();
    void random3D_ray_direction(); 


    void loaded(); 

    void expected_miss(); 
    void expected_hit(); 
    void attempt_to_tickle_axial_issue(); 

    bool match(const char* q) const ; 
    void method(); 
    bool is_intersect() const ; 


    std::string descGeom() const ; 
    std::string descRay() const ; 
    std::string descA() const ; 
    std::string descB() const ; 
    std::string descFull() const ; 
    std::string desc_() const ; 
    std::string desc() const ; 

    static std::string DescRay(const float3& o, const float3& v, const float t_min ); 
    static std::string DescIsect(const float4& isect, const float3& pos, const float sd ); 
    static std::string Desc(const float& v); 
    static std::string Desc(const float3& v); 
    static std::string Desc(const float4& v); 
}; 

CSGIntersectComparisonTest::CSGIntersectComparisonTest( const CSGNode* a_, const CSGNode* b_ ) 
    : 
    METHOD(SSys::getenvvar("METHOD", "loaded")),
    FULL(SSys::getenvbool("FULL")),
    RERUN(SSys::getenvvar("RERUN", "/tmp/simtrace.npy")),
    rerun( NP::LoadIfExists(RERUN) ),
    rr( rerun ? (const quad4*)rerun->values<float>() : nullptr ),
    AXES(SSys::getenvvar("AXES", "XZ")),
    h( strlen(AXES) > 0 ? AXES[0] - 'X' : ~0u ),
    v( strlen(AXES) > 1 ? AXES[1] - 'X' : ~0u ),
    o( OtherAxis(h,v) ),
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
    a_sd(0.f),
    b_sd(0.f),
    sd_winner(0u),
    t_min(SSys::getenvint("TMIN",0.f)),
    ray_origin(   SSys::getenvfloat3("RAYORI","0,0,0")), 
    ray_direction(SSys::getenvfloat3norm("RAYDIR","0,0,1")),
    oo((float*)&ray_origin),
    vv((float*)&ray_direction),
    seed(SSys::getenvunsigned("SEED", 1u)),
    rng(seed),
    margin(SSys::getenvfloat("MARGIN",1.2f)),
    x0(mn[X]*margin),
    x1(mx[X]*margin),
    y0(mn[Y]*margin),
    y1(mx[Y]*margin),
    z0(mn[Z]*margin),
    z1(mx[Z]*margin),
    h0(mn[h]*margin), 
    h1(mx[h]*margin), 
    v0(mn[v]*margin), 
    v1(mx[v]*margin), 
    o0(mn[o]*margin), 
    o1(mx[o]*margin), 
    num(SSys::getenvunsigned("NUM", 1000))
{
    init(); 
}

void CSGIntersectComparisonTest::init()
{ 
    assert( a_mn.x == b_mn.x ); 
    assert( a_mn.y == b_mn.y ); 
    assert( a_mn.z == b_mn.z );

    assert( a_mx.x == b_mx.x ); 
    assert( a_mx.y == b_mx.y ); 
    assert( a_mx.z == b_mx.z );

    sd_winner_stats[0u] = 0u ; 
    sd_winner_stats[1u] = 0u ; 
    sd_winner_stats[2u] = 0u ; 
}

std::string CSGIntersectComparisonTest::descStats() const 
{
    std::vector<std::string> labels = {{
        "sd_winner_stats[0u] A=B (draw  )", 
        "sd_winner_stats[1u] A<B (A wins)",
        "sd_winner_stats[2u] A>B (B wins)",
        "                         TOTAL: "
        }} ; 


    unsigned total = 0u ;   
    for(unsigned i=0 ; i < 3 ; i++ ) total += sd_winner_stats.at(i) ; 

    std::stringstream ss ; 
    ss << "descStats" << std::endl  ; 

    for(unsigned i=0 ; i < 3 ; i++)
    {
        ss << std::setw(30) << labels[i] 
           << std::setw(10) << sd_winner_stats.at(i)
           << std::fixed << std::setw(10) << std::setprecision(4) << float(sd_winner_stats.at(i))/float(total) 
           << std::endl
           ; 
    }
    ss << std::setw(30) << labels[3]
       << std::setw(10) << total 
       << std::endl 
       ; 

    std::string s = ss.str(); 
    return s ; 
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

unsigned CSGIntersectComparisonTest::smaller_sd() const
{
    float abs_a = std::abs(a_sd) ; 
    float abs_b = std::abs(b_sd) ; 
    return abs_a == abs_b ? 0u : ( abs_a < abs_b ? 1u : 2u ) ; 
} 

void CSGIntersectComparisonTest::intersect()
{
    zero(); 

    bool dump = false ; 
    a_valid_isect = intersect_leaf( a_isect, a , nullptr, nullptr, t_min, ray_origin, ray_direction, dump ); 
    b_valid_isect = intersect_leaf( b_isect, b , nullptr, nullptr, t_min, ray_origin, ray_direction, dump ); 

    status = ( int(a_valid_isect) << 1 ) | int(b_valid_isect) ; 

    // 0: A_MISS B_MISS  
    // 1: A_MISS B_HIT  
    // 2: A_HIT  B_MISS   
    // 3: A_HIT  B_HIT

    a_pos = a_valid_isect ? ray_origin + a_isect.w*ray_direction : zero3 ;  
    b_pos = b_valid_isect ? ray_origin + b_isect.w*ray_direction : zero3 ;  

    a_sd = a_valid_isect ? distance_leaf(a_pos, a, nullptr, nullptr ) : -1.f ;  
    b_sd = b_valid_isect ? distance_leaf(b_pos, b, nullptr, nullptr ) : -1.f ;  
    sd_winner = smaller_sd() ;  
    if( a_valid_isect && b_valid_isect ) sd_winner_stats[sd_winner] += 1u ;

    expected = is_expected();   // needs a_pos b_pos
    if(expected == false || is_intersect()) std::cout << desc() << std::endl; 


    quad4 sta ; 
    FormSimtrace(sta, a_isect, a_pos, t_min, ray_origin, ray_direction ); 
    a_simtrace.push_back(sta); 

    quad4 stb ; 
    FormSimtrace(stb, b_isect, b_pos, t_min, ray_origin, ray_direction ); 
    b_simtrace.push_back(stb); 

}

void CSGIntersectComparisonTest::save()
{
    LOG(info) << std::endl << descGeom() ; 
    LOG(info) << std::endl << descStats() ; 

    const char* fold = SPath::Resolve(FOLD, DIRPATH); 
    LOG(info) << fold ; 
    NP::Write(fold, "a_simtrace.npy",  (float*)a_simtrace.data(), a_simtrace.size(), 4, 4 ); 
    NP::Write(fold, "b_simtrace.npy",  (float*)b_simtrace.data(), b_simtrace.size(), 4, 4 ); 
}


void CSGIntersectComparisonTest::LoadSimtrace( const quad4& st, float& tmin, float3& ori, float3& dir )
{
    tmin = st.q1.f.w ; 
    ori.x = st.q2.f.x ; 
    ori.y = st.q2.f.y ; 
    ori.z = st.q2.f.z ; 

    dir.x = st.q3.f.x ; 
    dir.y = st.q3.f.y ; 
    dir.z = st.q3.f.z ; 
}

void CSGIntersectComparisonTest::FormSimtrace( quad4& st, const float4& isect, const float3& pos, float tmin, const float3& ori, const float3& dir )
{
    st.q0.f = isect ; 

    st.q1.f.x = pos.x ; 
    st.q1.f.y = pos.y ; 
    st.q1.f.z = pos.z ;
    st.q1.f.w = tmin ;

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
    ss << " A " << ( a_valid_isect ? DescIsect(a_isect, a_pos, a_sd) : "MISS" ) ;     
    std::string s = ss.str();
    return s ; 
}
std::string CSGIntersectComparisonTest::descB() const 
{
    std::stringstream ss ; 
    ss << " B " << ( b_valid_isect ? DescIsect(b_isect, b_pos, b_sd) : "MISS" ) ;     
    std::string s = ss.str();
    return s ; 
}
std::string CSGIntersectComparisonTest::descFull() const
{
    std::stringstream ss ; 
    ss << descRay() << std::endl 
       << descA() << std::endl 
       << descB() << std::endl 
       << descDiff() ; 
    std::string s = ss.str();
    return s ; 
}

std::string CSGIntersectComparisonTest::desc_() const
{
    std::stringstream ss ; 
    ss << descRay() << descA() << descDiff() ; 
    std::string s = ss.str();
    return s ; 
}
std::string CSGIntersectComparisonTest::desc() const
{
    return FULL ? descFull() : desc_() ; 
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
std::string CSGIntersectComparisonTest::DescIsect(const float4& isect, const float3& pos, const float sd ) // static 
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
       << ";"
       << std::scientific << sd 
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
    float a_mn = mn[axis]*1.2f ; 
    float a_mx = mx[axis]*1.2f ; 
    float a_st = (a_mx - a_mn)/float(num) ;   

    for(float a=a_mn ; a <= a_mx ; a += a_st )
    {
        oo[axis] = a ; 
        intersect(); 
    }
}

unsigned CSGIntersectComparisonTest::OtherAxis(unsigned h, unsigned v) // static
{
    assert( h != v ); 
    assert( h < 3 ); 
    assert( v < 3 ); 
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

void CSGIntersectComparisonTest::random2D()
{
    for(unsigned i=0 ; i < num ; i++ )
    {
        random2D_ray_origin();
        random2D_ray_direction();
        intersect(); 
    }
}
void CSGIntersectComparisonTest::random2D_fixed_origin()
{
    for(unsigned i=0 ; i < num ; i++ )
    {
        random2D_ray_direction();
        intersect(); 
    }
}

void CSGIntersectComparisonTest::random2D_ray_origin()
{
    double u0 = rng() ; 
    double u1 = rng() ; 
    float ah = h0 + u0*(h1 - h0 ); 
    float av = v0 + u1*(v1 - v0 ); 
    float ao = (o0 + o1)/2.f  ; 
    oo[h] = ah ; 
    oo[v] = av ; 
    oo[o] = ao ; 
}
void CSGIntersectComparisonTest::random2D_ray_direction()
{
    double phi, sinPhi, cosPhi ; 

    double u2 = rng() ; 
    phi = 2.*M_PIf*u2 ;     // azimuthal 0->2pi 
    ssincos(phi,sinPhi,cosPhi);  

    vv[h] = cosPhi ; 
    vv[v] = sinPhi ; 
    vv[o] = 0.f ; 
}


void CSGIntersectComparisonTest::random3D()
{
    for(unsigned i=0 ; i < num ; i++ )
    {
        random3D_ray_origin();
        random3D_ray_direction();
        intersect(); 
    }
}

void CSGIntersectComparisonTest::loaded()
{
    assert( rerun ); 
    LOG(info) << " rerun " << rerun->sstr() ; 
    num = rerun->shape[0] ; 
    for(unsigned i=0 ; i < num ; i++ )
    {
        const quad4& st = rr[i] ; 
        LoadSimtrace( st, t_min, ray_origin, ray_direction ); 
        intersect(); 
    }
}

void CSGIntersectComparisonTest::random3D_ray_origin()
{
    double u0 = rng() ; 
    double u1 = rng() ; 
    double u2 = rng() ; 

    oo[X] = x0 + u0*(x1 - x0 ); 
    oo[Y] = y0 + u1*(y1 - y0 );
    oo[Z] = z0 + u2*(z1 - z0 );
}
void CSGIntersectComparisonTest::random3D_ray_direction()
{
    double phi, sinPhi, cosPhi  ;
    double cosTheta, sinTheta ;  

    double u3 = rng() ; 
    double u4 = rng() ; 

    cosTheta = u3 ; 
    sinTheta = sqrtf(1.0-cosTheta*cosTheta);

    phi = 2.*M_PIf*u4 ;     // azimuthal 0->2pi 
    ssincos(phi,sinPhi,cosPhi);  

    vv[X] = sinTheta * cosPhi  ; 
    vv[Y] = sinTheta * sinPhi  ; 
    vv[Z] = cosTheta  ; 
}

void CSGIntersectComparisonTest::expected_miss()
{
    ray_origin = make_float3( 110.f, 0.f, 40.f ); 
    ray_direction = normalize(make_float3(0.993f,0.f, 0.119f)) ; 
    t_min = 0.f ; 
    intersect(); 
}
void CSGIntersectComparisonTest::expected_hit()
{
    ray_origin    = make_float3( 0.f, 0.f, 0.f ); 
    ray_direction = normalize(make_float3( 1.f, 1.f, 1.f) ); 
    intersect(); 
}
void CSGIntersectComparisonTest::attempt_to_tickle_axial_issue()
{
    ray_origin.z = -100.f ; 
    ray_origin.x = -264.f ; 
    random2D_fixed_origin() ; 
}

bool CSGIntersectComparisonTest::match(const char* q) const 
{
    return strcmp(METHOD, q) == 0 ; 
}
void CSGIntersectComparisonTest::method()
{
    LOG(info) << " METHOD " << METHOD ; 

    if(match("scanX"))         scan(X) ; 
    if(match("scanY"))         scan(Y) ; 
    if(match("scanZ"))         scan(Z) ; 
    if(match("random2D"))      random2D() ; 
    if(match("random3D"))      random3D() ; 
    if(match("expected_miss")) expected_miss() ; 
    if(match("expected_hit"))  expected_hit() ; 
    if(match("loaded"))        loaded() ; 
    if(match("intersect"))     intersect() ; 
}

bool CSGIntersectComparisonTest::is_intersect() const 
{
    return match("intersect") ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    float radius = SSys::getenvfloat("RADIUS", 264.f)  ; 
    float hz = SSys::getenvfloat("HZ", 0.15f ); 
    float z2 = hz ; 
    float z1 = -hz ; 
    CSGNode a_nd = CSGNode::OldCylinder( radius, z1, z2 ) ; 
    CSGNode b_nd = CSGNode::Cylinder(    radius, z1, z2 ) ; 
    CSGNode* a = &a_nd ; 
    CSGNode* b = &b_nd ; 


    CSGIntersectComparisonTest t(a, b); 
    if(t.rerun == nullptr) return 0 ; 

    t.method();     
    t.save(); 

    return 0 ; 
}



