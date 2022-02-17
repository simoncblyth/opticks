#include "scuda.h"
#include "sqat4.h"
#include "NP.hh"

#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGScan.h"

CSGScan::CSGScan( const char* dir_, const CSGFoundry* foundry_, const CSGSolid* solid_ ) 
    :
    dir(strdup(dir_)),
    foundry(foundry_),
    prim0(foundry->getPrim(0)),
    node0(foundry->getNode(0)),
    plan0(foundry->getPlan(0)),
    itra0(foundry->getItra(0)),
    solid(solid_),
    primIdx0(solid->primOffset),
    primIdx1(solid->primOffset+solid->numPrim)
{
}

void CSGScan::record(bool valid_isect, const float4& isect,  const float3& ray_origin, const float3& ray_direction )
{
    quad4 rec ;  

    rec.q0.f = make_float4(ray_origin);     rec.q0.i.w = int(valid_isect) ; 
    rec.q1.f = make_float4(ray_direction); // .w spare
    rec.q2.f = valid_isect ?  make_float4( ray_origin + isect.w*ray_direction, isect.w )  : make_float4(0.f) ; 
    rec.q3.f = isect ; 

    recs.push_back(rec);  
    //dump(rec); 
}


void CSGScan::trace(const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    for(unsigned primIdx=primIdx0 ; primIdx < primIdx1 ; primIdx++)
    {
        const CSGPrim* prim = prim0 + primIdx ;  
        //int numNode = prim->numNode(); 
        int nodeOffset = prim->nodeOffset(); 
        const CSGNode* node = node0 + nodeOffset ; 
        
        float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ; 
        bool valid_isect = intersect_prim(isect, node, plan0, itra0, t_min, ray_origin, ray_direction );
        record(valid_isect, isect, ray_origin, ray_direction );  
    } 
}


void CSGScan::dump( const quad4& rec )  // stat
{
    bool valid_isect = rec.q0.i.w == 1 ; 

    const float4& isect = rec.q3.f ; 
    const float4& ray_origin  = rec.q0.f ; 
    const float4& ray_direction = rec.q1.f ; 

    std::cout 
        << std::setw(30) << solid->label
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

    for(unsigned i=0 ; i < recs.size() ; i++)
    {
        const quad4& rec = recs[i] ; 
        bool hit = rec.q0.i.w == 1 ; 
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


void CSGScan::trace(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs )
{
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        const float3& ray_direction = dirs[i] ; 
        trace( t_min, ray_origin, ray_direction ); 
    }
}

void CSGScan::circle_scan()
{
    float t_min = 0.f ;
    float4 ce = solid->center_extent ; 
    float3 center = make_float3( ce ); 
    float extent = ce.w ; 
    float radius = 2.0f*extent ; 

    std::cout 
        << "CSGScan::circle_scan"
        << " extent " << extent 
        << " radius " << radius
        << std::endl 
        ;       

    // M_PIf from sutil_vec_math.h
    for(float phi=0. ; phi <= M_PIf*2.0 ; phi+=M_PIf*2.0/1000.0 )
    {
        float3 origin = center + make_float3( radius*sin(phi), 0.f, radius*cos(phi) ); 
        float3 direction = make_float3( -sin(phi),  0.f, -cos(phi) ); 
        trace(t_min, origin, direction );     
    }
    save("circle_scan");  
}


void CSGScan::_rectangle_scan(float t_min, unsigned n, float halfside, float y )
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

        trace(t_min, z_top, z_down );     
        trace(t_min, z_bot, z_up   );     

        x_lhs.z = v ; 
        x_rhs.z = v ; 
        trace(t_min, x_lhs, x_right );     
        trace(t_min, x_rhs, x_left  );     
    }
}

void CSGScan::rectangle_scan()
{
    float4 ce = solid->center_extent ; 
    float extent = ce.w ; 
    float halfside = 2.0f*extent ; 
    unsigned nxz = 100 ; 
    unsigned ny = 10 ; 
    float t_min = 0.f ;

    for(float y=-halfside ; y <= halfside ; y += halfside/float(ny) )
    {
        _rectangle_scan( t_min, nxz, halfside,   y );  
    }
    save("rectangle_scan"); 
}

void CSGScan::axis_scan()
{
    float t_min = 0.f ;
    float4 ce = solid->center_extent ; 
    float3 origin = make_float3(ce); 

    std::vector<float3> dirs ; 
    dirs.push_back( make_float3( 1.f, 0.f, 0.f));
    dirs.push_back( make_float3( 0.f, 1.f, 0.f));
    dirs.push_back( make_float3( 0.f, 0.f, 1.f));

    dirs.push_back( make_float3(-1.f, 0.f, 0.f));
    dirs.push_back( make_float3( 0.f,-1.f, 0.f));
    dirs.push_back( make_float3( 0.f, 0.f,-1.f));

    trace(t_min, origin, dirs );     

    save("axis_scan"); 
}

void CSGScan::save(const char* sub)
{
    //std::cout << " recs.size " << recs.size() << std::endl ; 
    if(recs.size() == 0 ) return ; 

    std::string name(solid->label,4) ;  // not NULL terminated, so give label size
    name += ".npy" ; 

    std::stringstream ss ; 
    ss << dir << "/" << sub ; 
    std::string fold = ss.str(); 

    NP::Write( fold.c_str(), name.c_str(), (float*)recs.data(), recs.size(), 4, 4 ) ; 
    recs.clear(); 
}



