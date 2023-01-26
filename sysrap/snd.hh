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

* how about convexpolyhedron with planes ? just add spl ? 
* how about polycone ?
* how about multi union ?

**/

#include <string>
#include "glm/fwd.hpp"
#include "SYSRAP_API_EXPORT.hh"

struct spa ; 
struct sbb ; 
struct sxf ; 
struct scsg ; 
struct NPFold ; 

struct SYSRAP_API snd
{
    static constexpr const char* NAME = "snd" ; 
    static constexpr const double zero = 0. ; 
    static scsg* POOL ; 
    static void SetPOOL( scsg* pool ); 
    static NPFold* Serialize(); 
    static void    Import(const NPFold* fold); 
    static std::string Desc();

    static const snd* GetND(  int idx);
    static const spa* GetPA( int idx);
    static const sxf* GetXF( int idx);
    static const sbb* GetBB(  int idx);

    static int GetNDXF(int idx) ; 

    static snd* GetND_(int idx);
    static spa* GetPA_(int idx);
    static sxf* GetXF_(int idx);
    static sbb* GetBB_(int idx);

    static std::string DescND(int idx);
    static std::string DescPA(int idx);
    static std::string DescXF(int idx);
    static std::string DescBB(int idx);

    static int Add(const snd& nd); 

    static constexpr const int N = 6 ;
    int tc ;  // typecode
    int fc ;  // first_child
    int nc ;  // num_child  
    int pa ;  // ref param
    int bb ;  // ref bbox
    int xf ;  // ref transform 

    void init(); 
    void setTC( int tc ); 
    void setPA( double x,  double y,  double z,  double w,  double z1, double z2 ); 
    void setBB( double x0, double y0, double z0, double x1, double y1, double z1 );
    void setXF( const glm::tmat4x4<double>& t ); 

    std::string brief() const ; 
    std::string desc() const ; 

    // signatures need to match CSGNode where these param will end up 
    static snd Sphere(double radius); 
    static snd ZSphere(double radius, double z1, double z2); 
    static snd Box3(double fullside); 
    static snd Box3(double fx, double fy, double fz ); 
    static snd Boolean(int op, int l, int r ); 
};


