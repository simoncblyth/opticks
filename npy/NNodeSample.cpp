#include "SSys.hh"
#include "NNodeSample.hpp"
#include "NNode.hpp"

#include "NPrimitives.hpp"


void NNodeSample::Tests(std::vector<nnode*>& nodes )
{
    // using default copy ctor to create nnode on heap

    nsphere* a = new nsphere(make_sphere(0.f,0.f,-50.f,100.f));
    nsphere* b = new nsphere(make_sphere(0.f,0.f, 50.f,100.f));
    nbox*    c = new nbox(make_box(0.f,0.f,0.f,200.f));

    nunion* u = new nunion(nunion::make_union( a, b ));
    nintersection* i = new nintersection(nintersection::make_intersection( a, b )); 
    ndifference* d1 = new ndifference(ndifference::make_difference( a, b )); 
    ndifference* d2 = new ndifference(ndifference::make_difference( b, a )); 
    nunion* u2 = new nunion(nunion::make_union( d1, d2 ));

    // reusing the primitives in multiple trees is a bit messy 

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





