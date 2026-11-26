/**
csg_intersect_leaf_test.cc
===========================

::

    ~/o/CSG/tests/csg_intersect_leaf_test.sh


1. creates single CSGNode sphere
2. setup environment for intersect_leaf call
3. does simtrace circle_scan with and without normalization
4. intersects saved into NPFold for python analysis


csg_intersect_leaf.h
   original impl that works with CSGNode only

csg_intersect_leaf_alt.h
    node traits template function intended to enable use
    with both CSGNode and srpn nodes without code duplication


**/

#include <cstdio>

#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#include "csg_intersect_leaf.h"

#ifdef WITH_CSG_ALT
#include "sparm.h"
#include "csg_intersect_leaf_alt.h"
#include "s_RPN.h"
#endif

#include "CSGNode.h"
#include "NPFold.h"

struct Geometry
{
    CSGNode nd ;
    const float4* plan ;
    const CSGNode* node ;
#ifdef WITH_CSG_ALT
    s_RPN* rpn ;
#endif

    std::vector<qat4>  tran ;
    std::vector<qat4>  itra ;
    std::vector<sparm> parm ;

    Geometry(float z_scale=0.5f);

    bool simtrace( quad4& p, bool norm ) const ;
    NP* circle_scan(bool norm) const ;
};


inline Geometry::Geometry(float z_scale)
    :
    nd(CSGNode::Sphere(100.f))
    ,plan(nullptr)
    ,node(&nd)
#ifdef WITH_CSG_ALT
    ,rpn(new s_RPN)
#endif
{
    std::array<float,16> ta = { 1.f , 0.f, 0.f, 0.f,
                                0.f , 1.f, 0.f, 0.f,
                                0.f , 0.f, z_scale, 0.f,
                                0.f , 0.f, 0.f, 1.f } ;

    std::array<float,16> va = { 1.f , 0.f, 0.f,  0.f,
                                0.f , 1.f, 0.f,  0.f,
                                0.f , 0.f, 1.f/z_scale, 0.f,
                                0.f , 0.f, 0.f,  1.f } ;

    qat4 t(ta.data()) ;
    qat4 v(va.data()) ;

    tran.push_back(t);
    itra.push_back(v);

    nd.setTransform( 1u + 0u );   // 0u means none

    unsigned typecode = nd.typecode();
    bool complement = nd.is_complement();
    bool identity = t.is_identity();
    int lvid = 0 ;

#ifdef WITH_CSG_ALT
    sparm pa = {} ;
    pa.q0 = nd.q0 ;
    pa.q1 = nd.q1 ;
    rpn->add_primnode( typecode, complement, identity, pa, v, lvid );
#endif


}

/**
Geometry::simtrace
-------------------

Follow quad4 layout from CSGQuery::simtrace

Notice no mystery regarding access to the transforms, the itra is
simply an argument to the intersect_leaf call.

**/

bool Geometry::simtrace( quad4& p, bool norm ) const
{
    float3* ray_origin = p.v2() ;
    float3* ray_direction = p.v3() ;
    float t_min = p.q1.f.w ;

    // the 1st float4 argumnent gives surface normal at intersect and distance

    float4& isect = p.q0.f ;
    bool valid_isect = false ;
    bool dumpxyz = false ;

#ifdef WITH_CSG_ALT

    {
        //intersect_leaf_alt<CSGNode_traits>(valid_isect, isect, node, plan, itra.data(), parm.data(), t_min, *ray_origin, *ray_direction, dumpxyz ) ;
    }

    {
        srpn*    rpn_node = rpn->node.data();
        float4*  rpn_plan = rpn->plan.data();
        qat4*    rpn_itra = rpn->itra.data();
        sparm*   rpn_parm = rpn->parm.data();

        intersect_leaf_alt<srpn_traits>(valid_isect, isect, rpn_node, rpn_plan, rpn_itra, rpn_parm, t_min, *ray_origin, *ray_direction, dumpxyz ) ;
    }

#else
    intersect_leaf(valid_isect, isect, node, plan, itra.data(), t_min, *ray_origin, *ray_direction, dumpxyz ) ;
#endif


    if( valid_isect )
    {
        if(norm)
        {
            float3* nrm = p.v0();
            *nrm = normalize(*nrm);
        }

        float t = p.q0.f.w ;
        float3 ipos = (*ray_origin) + t*(*ray_direction) ;
        p.q1.f.x = ipos.x ;
        p.q1.f.y = ipos.y ;
        p.q1.f.z = ipos.z ;
        //p.q1.f.w = distance(ipos) ;     // HMM: overwrite of tmin is problematic
    }
    return valid_isect ;
}


NP* Geometry::circle_scan(bool norm) const
{
    const int N = 360 ;
    NP* a = NP::Make<float>(N,4,4);
    quad4* aa = a->values<quad4>()  ;

    for(int i=0 ; i < N ; i++)
    {
        quad4& q = aa[i] ;

        float phi = M_PIf*float(i)/180.f ;

        float& t_min = q.q1.f.w ;
        float3* ori = q.v2() ;
        float3* dir = q.v3() ;

        t_min = 0.f ;

        ori->x = 0.f ;
        ori->y = 0.f ;
        ori->z = 0.f ;

        dir->x = cos(phi) ;
        dir->y = 0.f ;
        dir->z = sin(phi) ;

        simtrace(q, norm);
    }
    return a ;
}


void test_intersect_leaf()
{
    float z_scale = 0.5f ;
    //float z_scale = 1.0f ;

    Geometry g(z_scale) ;

    NPFold* f = new NPFold ;
    f->add("a", g.circle_scan(false) );
    f->add("b", g.circle_scan(true) );
    f->save("$FOLD");
}


int main()
{
    test_intersect_leaf() ;

    return 0 ;
}
