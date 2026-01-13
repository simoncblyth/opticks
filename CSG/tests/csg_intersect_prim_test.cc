
#include "NPFold.h"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "ssys.h"

#include "CSGNode.h"

#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

struct Params
{
    CSGNode*   node ;
    float4*    plan ;
    qat4*      tran ;
    qat4*      itra ;
    float      t_min ;
    bool      dumpxyz ;
};


struct csg_intersect_prim_test
{
    static int Main();
    static int Intersect(bool& valid_isect, float4& isect, CSGNode* nd, float t_min, const float3 o, const float3 d);
    static bool Simtrace( CSGNode* nd, quad4& q );
    static NP* XY(CSGNode* nd, int nx, int ny, float x0, float x1, float y0, float y1, float z, float _t_min, float3 d);

    static int One(CSGNode* nd, float t_min, const float3 o, const float3 d);
    static int SphereOne();
    static int CylinderOne();
    static int HalfSpaceOne();
    static int HalfCylinderOne();
    static int HalfCylinderXY();
};

inline int csg_intersect_prim_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "SphereOne");
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"SphereOne")) rc += SphereOne();
    if(ALL||0==strcmp(TEST,"CylinderOne")) rc += CylinderOne();
    if(ALL||0==strcmp(TEST,"HalfSpaceOne")) rc += HalfSpaceOne();
    if(ALL||0==strcmp(TEST,"HalfCylinderOne")) rc += HalfCylinderOne();
    if(ALL||0==strcmp(TEST,"HalfCylinderXY")) rc += HalfCylinderXY();
    return rc ;
}


inline int csg_intersect_prim_test::Intersect(bool& valid_isect, float4& isect, CSGNode* nd, float t_min, const float3 o, const float3 d)
{
    Params params = {} ;
    isect.x = 0.f ;
    isect.y = 0.f ;
    isect.z = 0.f ;
    isect.w = 0.f ;
    valid_isect = intersect_prim(isect, nd, params.plan, params.itra, t_min , o, d, params.dumpxyz );
    return 0 ;
}


inline int csg_intersect_prim_test::One(CSGNode* nd, float t_min, const float3 o, const float3 d)
{
    float4 isect ;
    bool valid_isect ;
    Intersect(valid_isect, isect, nd, t_min, o, d );
    printf("// o = np.array([%10.5f,%10.5f,%10.5f]) ; d = np.array([%10.5f,%10.5f,%10.5f]) ; is = %d ; isect = np.array([%10.5f,%10.5f,%10.5f,%10.5f]) \n",
          o.x, o.y, o.z,
          d.x, d.y, d.z,
          valid_isect,
          isect.x, isect.y, isect.z, isect.w ) ;
    return 0 ;
}
inline int csg_intersect_prim_test::SphereOne()
{
    CSGNode nd = CSGNode::Sphere(100.f);
    return One(&nd, 0.f, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f} );
}
inline int csg_intersect_prim_test::CylinderOne()
{
    CSGNode nd = CSGNode::Cylinder(100.f, -50.f, 50.f);
    return One(&nd, 0.f, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f} );
}
inline int csg_intersect_prim_test::HalfSpaceOne()
{
    float v = 1.f/sqrtf(2.f);
    CSGNode nd = CSGNode::HalfSpace(v,v,0.f,0.f);
    int rc = 0 ;
    rc += One(&nd, 0.f, { 1.f,  1.f, 100.f}, {0.f, 0.f, -1.f} );
    rc += One(&nd, 0.f, {-1.f, -1.f, 100.f}, {0.f, 0.f, -1.f} );
    return rc ;
}

inline int csg_intersect_prim_test::HalfCylinderOne()
{
    CSGNode* nd = CSGNode::HalfCylinder();
    int rc = 0 ;
    rc += One(nd, 0.f, { 1.f,  1.f, 100.f}, {0.f, 0.f, -1.f} );
    rc += One(nd, 0.f, {-1.f, -1.f, 100.f}, {0.f, 0.f, -1.f} );
    return rc ;
}



inline NP* csg_intersect_prim_test::XY(CSGNode* nd, int nx, int ny, float x0, float x1, float y0, float y1, float z, float _t_min, float3 ray_direction )
{
    const int N = nx*ny ;
    NP* xy = NP::Make<float>(N,4,4);
    quad4* aa = xy->values<quad4>()  ;

    for(int iy=0 ; iy < ny ; iy++)
    for(int ix=0 ; ix < nx ; ix++)
    {
        int idx = iy*nx + ix ;
        assert( idx < N );

        float fx = float(ix)/float(nx-1) ;
        float fy = float(iy)/float(ny-1) ;
        float x = x0 + (x1-x0)*fx ;
        float y = y0 + (y1-y0)*fy ;

        quad4& q = aa[idx] ;

        float& t_min = q.q1.f.w ;
        float3* o = q.v2() ;
        float3* d = q.v3() ;

        t_min = _t_min ;

        o->x = x ;
        o->y = y ;
        o->z = z ;

        d->x = ray_direction.x ;
        d->y = ray_direction.y ;
        d->z = ray_direction.z ;

        Simtrace(nd, q );
    }
    return xy ;
}

bool csg_intersect_prim_test::Simtrace( CSGNode* nd, quad4& q )
{
    float3* ray_origin = q.v2() ;
    float3* ray_direction = q.v3() ;
    float t_min = q.q1.f.w ;

    float4& isect = q.q0.f ;
    bool valid_isect = false ;
    bool dumpxyz = false ;
    Intersect(valid_isect, isect, nd, t_min, *ray_origin, *ray_direction ) ;
    if( valid_isect )
    {
        float t = isect.w ;
        float3 ipos = (*ray_origin) + t*(*ray_direction) ;
        q.q1.f.x = ipos.x ;
        q.q1.f.y = ipos.y ;
        q.q1.f.z = ipos.z ;
    }
    return valid_isect ;
}



/**
csg_intersect_prim_test::XY
----------------------------

               Z
               |
               |
          +----+----+
          |         |
          |         |
       ---+----0----+---  X
          |         |
          |         |
          +----+----+
               |
               |


HMM so far do not see issue of getting full circle with cxr_min.sh ?::

    MODE=3 CIRCLE=0,0,50,100 NCIRCLE=0,0,1 ~/o/CSG/tests/csg_intersect_prim_test.sh pdb

**/

inline int csg_intersect_prim_test::HalfCylinderXY()
{
    float radius = 100.f ;
    CSGNode* nd = nullptr ;
    {
        float v = 1.f/sqrtf(2.f);
        float z0 = -50.f ;
        float z1 =  50.f ;
        nd = CSGNode::HalfCylinder(v,v,0.f,0.f,  radius, z0, z1);
    }

    // HMM:bbox to pick scan range

    float oz = 100.f ;
    float3 d = {0.f, 0.f, -1.f} ;
    float t_min = 0.f ;

    int nx = 41 ;
    int ny = 41 ;

    float ext = radius*1.1f ;

    NP* xy = XY(nd, nx, ny, -ext, ext, -ext, ext, oz, t_min, d );

    std::cout << "csg_intersect_prim_test::HalfCylinderXY xy " << ( xy ? xy->sstr() : "-" ) << "\n" ;

    NPFold* f = new NPFold ;
    f->add("xy", xy);
    f->save("$FOLD/$TEST");

    return 0 ;
}

int main()
{
    return csg_intersect_prim_test::Main() ;
}


