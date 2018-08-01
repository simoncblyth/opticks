#include "SSys.hh"
#include "NNodeSample.hpp"
#include "NNode.hpp"

#include "NPrimitives.hpp"


nnode* NNodeSample::Sphere1()
{
    nsphere* a = new nsphere(make_sphere(0.f,0.f,-50.f,100.f));
    return a ; 
}
nnode* NNodeSample::Sphere2()
{
    nsphere* b = new nsphere(make_sphere(0.f,0.f, 50.f,100.f));
    return b ; 
}
nnode* NNodeSample::Union1()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    nunion* u = new nunion(nunion::make_union( a, b ));
    return u ; 
}
nnode* NNodeSample::Intersection1()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    nintersection* i = new nintersection(nintersection::make_intersection( a, b )); 
    return i ; 
}
nnode* NNodeSample::Difference1()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    ndifference* d1 = new ndifference(ndifference::make_difference( a, b )); 
    return d1 ; 
}
nnode* NNodeSample::Difference2()
{
    nnode* a = Sphere1() ; 
    nnode* b = Sphere2() ; 
    ndifference* d2 = new ndifference(ndifference::make_difference( b, a )); 
    return d2 ; 
}
nnode* NNodeSample::Union2()
{
    nnode* d1 = Difference1() ; 
    nnode* d2 = Difference2() ; 
    nunion* u2 = new nunion(nunion::make_union( d1, d2 ));
    return u2 ; 
}
nnode* NNodeSample::Box()
{
    nbox*    c = new nbox(make_box(0.f,0.f,0.f,200.f));
    return c ; 
}
nnode* NNodeSample::SphereBoxUnion()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = new nsphere(make_sphere(0.f,0.f,0.f,radius));
    nbox*    bx = new nbox(make_box(0.f,0.f,0.f, inscribe ));
    nunion*  u_sp_bx = new nunion(nunion::make_union( sp, bx ));

    return u_sp_bx ;
}
nnode* NNodeSample::SphereBoxIntersection()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = new nsphere(make_sphere(0.f,0.f,0.f,radius));
    nbox*    bx = new nbox(make_box(0.f,0.f,0.f, inscribe ));
    nintersection*  i_sp_bx = new nintersection(nintersection::make_intersection( sp, bx ));

    return i_sp_bx ;
}
nnode* NNodeSample::SphereBoxDifference()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = new nsphere(make_sphere(0.f,0.f,0.f,radius));
    nbox*    bx = new nbox(make_box(0.f,0.f,0.f, inscribe ));
    ndifference*    d_sp_bx = new ndifference(ndifference::make_difference( sp, bx ));

    return d_sp_bx ;
}
nnode* NNodeSample::BoxSphereDifference()
{
    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = new nsphere(make_sphere(0.f,0.f,0.f,radius));
    nbox*    bx = new nbox(make_box(0.f,0.f,0.f, inscribe ));
    ndifference*    d_bx_sp = new ndifference(ndifference::make_difference( bx, sp ));

    return d_bx_sp ;
}

void NNodeSample::Tests(std::vector<nnode*>& nodes )
{
    nodes.push_back( Sphere1() ) ; 
    nodes.push_back( Sphere2() ) ; 
    nodes.push_back( Union1() ) ; 
    nodes.push_back( Intersection1() ) ; 
    nodes.push_back( Difference1() ) ; 
    nodes.push_back( Difference2() ) ; 
    nodes.push_back( Union2() ) ; 
    nodes.push_back( Box() ) ; 
    nodes.push_back( SphereBoxUnion() ) ; 
    nodes.push_back( SphereBoxIntersection() ) ; 
    nodes.push_back( SphereBoxDifference() ) ; 
    nodes.push_back( BoxSphereDifference() ) ; 
}

void NNodeSample::Tests_OLD_MESSY_WAY(std::vector<nnode*>& nodes )
{
    assert(0) ; 
    // Using the same primitive instance in multiple trees is messy
    // and causes failures from changing parent links  
    // SO DONT DO THAT 

    // using default copy ctor to create nnode on heap

    nsphere* a = new nsphere(make_sphere(0.f,0.f,-50.f,100.f));
    nsphere* b = new nsphere(make_sphere(0.f,0.f, 50.f,100.f));
    nbox*    c = new nbox(make_box(0.f,0.f,0.f,200.f));

    nunion* u = new nunion(nunion::make_union( a, b ));
    nintersection* i = new nintersection(nintersection::make_intersection( a, b )); 
    ndifference* d1 = new ndifference(ndifference::make_difference( a, b )); 
    ndifference* d2 = new ndifference(ndifference::make_difference( b, a )); 
    nunion* u2 = new nunion(nunion::make_union( d1, d2 ));
    nodes.push_back( (nnode*)a );
    nodes.push_back( (nnode*)b );
    nodes.push_back( (nnode*)u );
    nodes.push_back( (nnode*)i );
    nodes.push_back( (nnode*)d1 );
    nodes.push_back( (nnode*)d2 );
    nodes.push_back( (nnode*)u2 );

    nodes.push_back( (nnode*)c );

    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = new nsphere(make_sphere(0.f,0.f,0.f,radius));
    nbox*    bx = new nbox(make_box(0.f,0.f,0.f, inscribe ));

    nunion*  u_sp_bx = new nunion(nunion::make_union( sp, bx ));
    nintersection*  i_sp_bx = new nintersection(nintersection::make_intersection( sp, bx ));
    ndifference*    d_sp_bx = new ndifference(ndifference::make_difference( sp, bx ));
    ndifference*    d_bx_sp = new ndifference(ndifference::make_difference( bx, sp ));

    nodes.push_back( (nnode*)u_sp_bx );
    nodes.push_back( (nnode*)i_sp_bx );
    nodes.push_back( (nnode*)d_sp_bx );
    nodes.push_back( (nnode*)d_bx_sp );
}

nnode* NNodeSample::_Prepare(nnode* root)
{
    root->update_gtransforms();
    root->verbosity = SSys::getenvint("VERBOSITY", 1) ; 
    root->dump() ; 
    const char* boundary = "Rock//perfectAbsorbSurface/Vacuum" ;
    root->set_boundary(boundary); 
    return root ; 
}
nnode* NNodeSample::DifferenceOfSpheres()
{
    nsphere* a = new nsphere(make_sphere( 0.000,0.000,0.000,500.000 )) ; a->label = "a" ;   
    nsphere* b = new nsphere(make_sphere( 0.000,0.000,0.000,100.000 )) ; b->label = "b" ;   
    ndifference* ab = new ndifference(ndifference::make_difference( a, b )) ; ab->label = "ab" ; a->parent = ab ; b->parent = ab ;  ;   
    return _Prepare(ab) ; 
}
nnode* NNodeSample::Box3()
{
    nbox* a = new nbox(make_box3(300,300,200)) ; 
    return _Prepare(a) ; 
}
nnode* NNodeSample::Sample(const char* name)
{
    if(strcmp(name, "DifferenceOfSpheres") == 0) return DifferenceOfSpheres();
    if(strcmp(name, "Box3") == 0)                return Box3();
    return NULL ; 
}





