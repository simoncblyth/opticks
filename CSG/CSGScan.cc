/**
CSGScan.cc
==========

Restructured this in preparaction for the CUDA/MOCK_CUDA impl
by preparing all the rays ahead of time and
then doing all the intersects together. 

TODO: split off the CUDA/MOCK_CUDA part that can be parallelized 


**/

#include "NPX.h"
#include "sstr.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGScan.h"

#include "CSGParams.h"


CSGScan::CSGScan( const CSGFoundry* fd_, const CSGSolid* so_, const char* opts_  ) 
    :
    fd(fd_),
    prim0(fd->getPrim(0)),
    node0(fd->getNode(0)),
    so(so_),
    primIdx0(so->primOffset),
    primIdx1(so->primOffset+so->numPrim),
    primIdx(primIdx0),   // 
    prim(prim0 + primIdx),
    nodeOffset(prim->nodeOffset()),
    node(node0 + nodeOffset),
    ctx(new CSGParams {})
{
    initGeom(); 
    initRays(opts_); 
}

void CSGScan::initGeom()
{
    ctx->node = node ; 
    ctx->plan = fd->getPlan(0) ; 
    ctx->itra = fd->getItra(0) ;
}

void CSGScan::initRays(const char* opts_)
{
    std::vector<std::string> opts ;
    sstr::Split(opts_, ',', opts ); 

    std::vector<quad4> qq ; 
    for(unsigned i=0 ; i < opts.size() ; i++)
    {
        const char* opt = opts[i].c_str(); 
        add_scan(qq, opt); 
    }

    ctx->num = qq.size() ; 
    ctx->qq = new quad4[ctx->num];  
    ctx->tt = new quad4[ctx->num];
}

void CSGScan::add_scan(std::vector<quad4>& qq, const char* opt)
{
    if(strcmp(opt,"axis")==0)   add_axis_scan(qq); 
    if(strcmp(opt,"circle")==0) add_circle_scan(qq); 
    if(strcmp(opt,"rectangle")==0) add_rectangle_scan(qq); 
}

void CSGScan::add_axis_scan(std::vector<quad4>& qq)
{
    float t_min = 0.f ;
    float4 ce = so->center_extent ; 
    float3 origin = make_float3(ce); 

    std::vector<float3> dirs ; 
    dirs.push_back( make_float3( 1.f, 0.f, 0.f));
    dirs.push_back( make_float3( 0.f, 1.f, 0.f));
    dirs.push_back( make_float3( 0.f, 0.f, 1.f));

    dirs.push_back( make_float3(-1.f, 0.f, 0.f));
    dirs.push_back( make_float3( 0.f,-1.f, 0.f));
    dirs.push_back( make_float3( 0.f, 0.f,-1.f));

    add_q(qq, t_min, origin, dirs );     
}

void CSGScan::add_circle_scan(std::vector<quad4>& qq)
{
    float t_min = 0.f ;
    float4 ce = so->center_extent ; 
    float3 center = make_float3( ce ); 
    float extent = ce.w ; 
    float radius = 2.0f*extent ; 

    std::cout 
        << "CSGScan::add_circle_scan"
        << " extent " << extent 
        << " radius " << radius
        << std::endl 
        ;       

    // M_PIf from sutil_vec_math.h
    for(float phi=0. ; phi <= M_PIf*2.0 ; phi+=M_PIf*2.0/1000.0 )
    {
        float3 origin = center + make_float3( radius*sin(phi), 0.f, radius*cos(phi) ); 
        float3 direction = make_float3( -sin(phi),  0.f, -cos(phi) ); 
        add_q(qq, t_min, origin, direction);     
    }
}

void CSGScan::add_rectangle_scan(std::vector<quad4>& qq)
{
    float4 ce = so->center_extent ; 
    float extent = ce.w ; 
    float halfside = 2.0f*extent ; 
    unsigned nxz = 100 ; 
    unsigned ny = 10 ; 
    float t_min = 0.f ;

    for(float y=-halfside ; y <= halfside ; y += halfside/float(ny) )
    {
        _add_rectangle_scan(qq, t_min, nxz, halfside,   y );  
    }
}

void CSGScan::_add_rectangle_scan(std::vector<quad4>& qq, float t_min, unsigned n, float halfside, float y )
{
    // shooting up/down 

    float3 z_up   = make_float3( 0.f, 0.f,  1.f);
    float3 z_down = make_float3( 0.f, 0.f, -1.f);

    float3 z_top = make_float3( 0.f, y,  halfside ); 
    float3 z_bot = make_float3( 0.f, y, -halfside ); 

    // shooting left/right

    float3 x_right = make_float3(  1.f, 0.f,  0.f);
    float3 x_left  = make_float3( -1.f, 0.f,  0.f);

    float3 x_lhs = make_float3( -halfside, y,  0.f ); 
    float3 x_rhs = make_float3(  halfside, y,  0.f ); 

    for(float v=-halfside ; v <= halfside ; v+= halfside/float(n) )
    { 
        z_top.x = v ; 
        z_bot.x = v ; 

        add_q(qq, t_min, z_top, z_down );     
        add_q(qq, t_min, z_bot, z_up   );     

        x_lhs.z = v ; 
        x_rhs.z = v ; 
        add_q(qq, t_min, x_lhs, x_right );     
        add_q(qq, t_min, x_rhs, x_left  );     
    }
}

void CSGScan::add_q(std::vector<quad4>& qq, const float t_min, const float3& ray_origin, const std::vector<float3>& dirs )
{
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        const float3& ray_direction = dirs[i] ; 
        add_q(qq, t_min, ray_origin, ray_direction ); 
    }
}

void CSGScan::add_q(std::vector<quad4>& qq, float t_min, const float3& ray_origin, const float3& ray_direction )
{
    quad4 q = {} ;  

    q.q0.f = make_float4(ray_origin);  
    q.q1.f = make_float4(ray_direction);  
    q.q1.f.w = t_min ; 

    qq.push_back(q);  
}

void CSGScan::intersect_scan()
{
    for(int i=0 ; i < ctx->num ; i++)
    {
        ctx->intersect(i); 
    }
}


void CSGScan::dump( const quad4& t )  // stat
{
    bool valid_isect = t.q0.i.w == 1 ; 

    const float4& isect = t.q3.f ; 
    const float4& ray_origin  = t.q0.f ; 
    const float4& ray_direction = t.q1.f ; 

    std::cout 
        << std::setw(30) << so->label
        << " valid_isect " << valid_isect 
        << " isect ( "
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.x 
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.y
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.z 
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.w
        << " ) "
        << " dir ( "
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_direction.x 
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_direction.y
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_direction.z 
        << " ) "
        << " ori ( "
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.x 
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.y
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.z 
        << " ) "
        << std::endl 
        ; 
}

std::string CSGScan::brief() const
{
    unsigned nhit = 0 ; 
    unsigned nmiss = 0 ; 

    for(int i=0 ; i < ctx->num ; i++)
    {
        const quad4& q = ctx->qq[i] ; 
        const quad4& t = ctx->tt[i] ;
 
        bool hit = q.q0.i.w == 1 ; 
        if(hit)  nhit += 1 ; 
        if(!hit) nmiss += 1 ; 
    }
    std::stringstream ss ; 
    ss
        << " nhit " << nhit 
        << " nmiss " << nmiss 
        ;

    std::string s = ss.str() ; 
    return s ; 
}



NPFold* CSGScan::serialize() const
{
    NPFold* fold = new NPFold ; 
    NP* _qq = NPX::ArrayFromData<float>( (float*)ctx->qq, 4, 4 ) ;  
    NP* _tt = NPX::ArrayFromData<float>( (float*)ctx->tt, 4, 4 ) ;  
    fold->add("qq", _qq ); 
    fold->add("tt", _tt ); 
    return fold ;  
}

void CSGScan::save(const char* base, const char* sub) const 
{
   NPFold* fold = serialize();  
   fold->save(base, sub) ; 
}


