#pragma once
/**
snd.hh : constituent CSG node in preparation
============================================= 

snd.h intended as minimal first step, transiently holding 
parameters of G4VSolid CSG trees for subsequent use by CSGNode::Make
and providing dependency fire break between G4 and CSG 

* initially thought snd.h would be transient with no persisting
  and no role on GPU. But that is inconsistent with the rest of stree.h and also 
  want to experiment with non-binary intersection in future, 
  so can use snd.h as testing ground for non-binary solid persisting.  

* snd.h instances are one-to-one related with CSG/CSGNode.h, 

Usage requires initializing the static "pools"::

    #include "snd.hh"
    scsg* snd::POOL = new scsg ; 

TODO: 

* its problematic that on stack expts setting Param, AABB, XForm will write into the global pool  
* need control of pool in use + avoiding static initialization requirement, scsg.{hh,cc} ?
* how about convexpolyhedron with planes ?
* how about polycone ?
* how about multi union ?

**/

#include <string>

#include "OpticksCSG.h"
#include "scuda.h"
#include "stran.h"

#include "SYSRAP_API_EXPORT.hh"

struct spa ; 
struct sbb ; 
struct sxf ; 
struct scsg ; 

struct SYSRAP_API snd
{
    static constexpr const char* NAME = "snd" ; 
    static constexpr const double zero = 0. ; 
    static scsg* POOL ; 
    static void SetPOOL( scsg* pool ); 
    static int Add(const snd& nd); 

    static constexpr const int N = 6 ;
    int tc ;  // typecode
    int fc ;  // first_child
    int nc ;  // num_child  
    int pa ;  // ref param
    int bb ;  // ref bbox
    int xf ;  // ref transform 

    void init(); 
    void setTypecode( unsigned tc ); 
    void setParam( double x,  double y,  double z,  double w,  double z1, double z2 ); 
    void setAABB(  double x0, double y0, double z0, double x1, double y1, double z1 );
    void setXForm( const glm::tmat4x4<double>& t ); 

    std::string brief() const ; 
    std::string desc() const ; 

    // follow CSGNode where these param will end up 
    static snd Sphere(double radius); 
    static snd ZSphere(double radius, double z1, double z2); 
    static snd Box3(double fullside); 
    static snd Box3(double fx, double fy, double fz ); 
    static snd Boolean(OpticksCSG_t op,  int l, int r ); 
};


