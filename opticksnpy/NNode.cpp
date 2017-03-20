
#include <cstdio>
#include <cassert>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "NNode.hpp"
#include "NPart.hpp"
#include "NQuad.hpp"
#include "NSphere.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"


double nnode::operator()(double,double,double) 
{
    return 0.f ; 
} 

std::string nnode::desc()
{
    std::stringstream ss ; 
    ss  << " nnode "
        << std::setw(3) << type 
        << std::setw(15) << CSGName(type) 
        ;     
    return ss.str();
}


void nnode::dump(const char* msg)
{
    printf("(%s)%s\n",csgname(), msg);
    if(left && right)
    {
        left->dump("left");
        right->dump("right");
    }
}

void nnode::Init( nnode& n , OpticksCSG_t type, nnode* left, nnode* right )
{
    n.type = type ; 
    n.left = left ; 
    n.right = right ; 
    n.param = {0.f, 0.f, 0.f, 0.f };
}

const char* nnode::csgname()
{
   return CSGName(type);
}
unsigned nnode::maxdepth()
{
    return _maxdepth(0);
}
unsigned nnode::_maxdepth(unsigned depth)  // recursive 
{
    return left && right ? nmaxu( left->_maxdepth(depth+1), right->_maxdepth(depth+1)) : depth ;  
}


npart nnode::part()
{
    nbbox bb = bbox();

    npart pt ; 
    pt.zero();
    pt.setParam( param );
    pt.setTypeCode( type );
    pt.setBBox( bb );

    return pt ; 
}


nbbox nnode::bbox()
{
   // needs to be overridden for primitives
    nbbox bb = make_nbbox() ; 
    if(left && right)
    {
        bb.include( left->bbox() );
        bb.include( right->bbox() );
    }
    return bb ; 
}



double nunion::operator()(double px, double py, double pz) 
{
    assert( left && right );
    double l = (*left)(px, py, pz) ;
    double r = (*right)(px, py, pz) ;
    return fmin(l, r);
}
double nintersection::operator()(double px, double py, double pz) 
{
    assert( left && right );
    double l = (*left)(px, py, pz) ;
    double r = (*right)(px, py, pz) ;
    return fmax( l, r);
}
double ndifference::operator()(double px, double py, double pz) 
{
    assert( left && right );
    double l = (*left)(px, py, pz) ;
    double r = (*right)(px, py, pz) ;
    return fmax( l, -r);    // difference is intersection with complement, complement negates signed distance function
}


void nnode::Tests(std::vector<nnode*>& nodes )
{
    nsphere* a = make_nsphere_ptr(0.f,0.f,-50.f,100.f);
    nsphere* b = make_nsphere_ptr(0.f,0.f, 50.f,100.f);
    nbox*    c = make_nbox_ptr(0.f,0.f,0.f,200.f);

    nunion* u = make_nunion_ptr( a, b );
    nintersection* i = make_nintersection_ptr( a, b ); 
    ndifference* d1 = make_ndifference_ptr( a, b ); 
    ndifference* d2 = make_ndifference_ptr( b, a ); 
    nunion* u2 = make_nunion_ptr( d1, d2 );

    nodes.push_back( (nnode*)a );
    nodes.push_back( (nnode*)b );
    nodes.push_back( (nnode*)u );
    nodes.push_back( (nnode*)i );
    nodes.push_back( (nnode*)d1 );
    nodes.push_back( (nnode*)d2 );
    nodes.push_back( (nnode*)u2 );

    nodes.push_back( (nnode*)c );
}


