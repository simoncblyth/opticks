/**
csg_intersect_node_test.cc
===========================

* NB node includes leaf and listnode, excludes tree

::

    ~/o/CSG/tests/csg_intersect_node_test.sh


csg_intersect_node.h
    original impl that works with CSGNode only

csg_intersect_node_alt.h
    node traits template function intended to enable use
    with both CSGNode and srpn nodes without code duplication

**/

#include <cstdio>

#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#include "OpticksCSG.h"

#include "CSGNode.h"

#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"

#include "sparm.h"
#include "csg_intersect_leaf_alt.h"
#include "csg_intersect_node_alt.h"
#include "s_RPN.h"

#include "NPFold.h"





struct Geometry
{
    // HMM: would jump to CSGFoundry - but that brings in too much else ?
    // another reason for lite weight CSGFoundryModel.h  ?
    // splitting the involved population from underlying simple model

    const char* name ;
    s_RPN*      rpn ;

    std::vector<CSGNode> node ;
    std::vector<qat4>    tran ;
    std::vector<qat4>    itra ;
    std::vector<sparm>   parm ;
    std::vector<float4>  plan ;

    Geometry(const char* name);

    void init();
    void init_ellipsoid();
};


inline Geometry::Geometry(const char* name_)
    :
    name(name_ ? strdup(name_) : nullptr ),
    rpn(new s_RPN)
{
    init();
}

inline void Geometry::init()
{
    if(0==strcmp(name, "ellipsoid")) init_ellipsoid();
}


/**
Geometry::init_ellipsoid
--------------------------

It looks like duplication between CSGNode and srpn approaches
with simple geometry - but once get more complicated the
split becomes inevitable as node ordering and associated transform
ordering will be different between the approaches

**/

inline void Geometry::init_ellipsoid()
{
    CSGNode nd = CSGNode::Sphere(100.f) ;
    nd.setTransform( 1u + 0u );   // 0u means none

    unsigned typecode = nd.typecode();
    bool complement = nd.is_complement();

    node.push_back(nd);

    float z_scale = 0.5f ;

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

    bool identity = v.is_identity();
    int lvid = 0 ;

    sparm pa = {} ;
    pa.q0 = nd.q0 ;
    pa.q1 = nd.q1 ;
    rpn->add_primnode( typecode, complement, identity, pa, v, lvid );
}



struct IntersectTest
{
    IntersectTest(Geometry* geom);
    Geometry* geom ;

    bool simtrace( quad4& p, bool norm ) const ;
    NP* circle_scan(bool norm) const ;
};


inline IntersectTest::IntersectTest(Geometry* geom_)
   :
   geom(geom_)
{
}


/**
IntersectTest::simtrace
-------------------------

Follow quad4 layout from CSGQuery::simtrace

Notice no mystery regarding access to the transforms, the itra is
simply an argument to the intersect_leaf call.

**/

bool IntersectTest::simtrace( quad4& p, bool norm ) const
{
    float3* ray_origin = p.v2() ;
    float3* ray_direction = p.v3() ;
    float t_min = p.q1.f.w ;

    // the 1st float4 argumnent gives surface normal at intersect and distance

    bool dumpxyz = false ;

    std::array<float4,3> isect_ = {} ;
    std::array<bool,3>   valid_isect = {false, false, false} ;

    {
        CSGNode* node = geom->node.data();
        float4*  plan = geom->plan.data();
        qat4*    itra = geom->itra.data();
        sparm*   parm = nullptr ;

        intersect_node_alt<CSGNode_traits>(valid_isect[0], isect_[0], node, node, plan, itra, parm, t_min, *ray_origin, *ray_direction, dumpxyz ) ;
    }

    {
        srpn*    node = geom->rpn->node.data();
        qat4*    itra = geom->rpn->itra.data();
        sparm*   parm = geom->rpn->parm.data();
        float4*  plan = geom->rpn->plan.data();

        intersect_node_alt<srpn_traits>(valid_isect[1], isect_[1], node, node, plan, itra, parm, t_min, *ray_origin, *ray_direction, dumpxyz ) ;
    }

    {
        CSGNode* node = geom->node.data() ;
        float4* plan  = geom->plan.data() ;
        qat4*   itra  = geom->itra.data() ;

        intersect_node(valid_isect[2], isect_[2], node, node, plan, itra, t_min, *ray_origin, *ray_direction, dumpxyz ) ;
    }


    bool consistent = valid_isect[0] == valid_isect[1] && valid_isect[0] == valid_isect[2] ;
    assert( consistent );

    assert( isect_[0].x == isect_[1].x && isect_[0].x == isect_[2].x );
    assert( isect_[0].y == isect_[1].y && isect_[0].y == isect_[2].y );
    assert( isect_[0].z == isect_[1].z && isect_[0].z == isect_[2].z );
    assert( isect_[0].w == isect_[1].w && isect_[0].w == isect_[2].w );


    int pick = 0;

    float4& isect = p.q0.f ;
    isect = isect_[pick] ;

    if( valid_isect[pick] )
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
    return valid_isect[pick] ;
}


NP* IntersectTest::circle_scan(bool norm) const
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


int test_intersect_node()
{
    const char* TEST = ssys::getenvvar("TEST", "ellipsoid");
    Geometry geom(TEST) ;
    IntersectTest it(&geom);

    NPFold* f = new NPFold ;
    f->add("a", it.circle_scan(false) );
    f->add("b", it.circle_scan(true) );
    f->save("$FOLD");
    return 0 ;
}


int main()
{
    return test_intersect_node() ;
}
