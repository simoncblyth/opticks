
#include <cstdio>
#include <cassert>
#include <cmath>

#include "NNode.hpp"


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



