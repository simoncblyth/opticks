
#include <cstdio>
#include <cassert>
#include <cmath>

#include "NNode.hpp"
#include "NBBox.hpp"


double nnode::operator()(double,double,double) 
{
    return 0.f ; 
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

const char* nnode::csgname()
{
   return CSGName(type);
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



