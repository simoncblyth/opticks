#pragma once
/**
snd.hh : constituent CSG node in preparation
============================================= 

snd.h intended as minimal first step, holding parameters of 
G4VSolid CSG trees for subsequent use by CSGNode::Make
and providing dependency fire break between G4 and CSG 

* snd.h instances are one-to-one related to CSG/CSGNode.h

* initially thought snd.h would be transient with no persisting and no role on GPU. 
  But that is inconsistent with the rest of stree.h and also want to experiment 
  with non-binary intersection in future, so are using snd.h to test non-binary 
  solid persisting following the same approach as snode.h structural nodes

Usage requires the scsg.hh POOL. That is now done at stree instanciation::

    snd::SetPOOL(new scsg); 

TODO: 

* add polycone ZNudge and tests ?
* how about convexpolyhedron with planes ? just add spl ? 
* how about multiunion ?

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

    // HMM: maybe can remove these now that are doing scsg.hh POOL hookup with stree::init
    static NPFold* Serialize(); 
    static void    Import(const NPFold* fold); 

    static std::string Desc();

    static const snd* GetNode(  int idx);
    static const spa* GetParam( int idx);
    static const sxf* GetXForm( int idx);
    static const sbb* GetAABB(  int idx);

    static int  GetNodeXForm(int idx) ; 
    static void SetNodeXForm(int idx, const glm::tmat4x4<double>& tr );

    //static void SetLVID(int idx, int lvid);  // label node tree 

    static snd* GetNode_(int idx);
    static spa* GetParam_(int idx);
    static sxf* GetXForm_(int idx);
    static sbb* GetAABB_(int idx);

    static std::string Desc(int idx);
    static std::string DescParam(int idx);
    static std::string DescXForm(int idx);
    static std::string DescAABB( int idx);

    static int Add(const snd& nd); 



    int index ; 
    int depth ; 
    int sibdex ;  // 0-based sibling index 
    int parent ; 

    int num_child ; 
    int first_child ; 
    int next_sibling ; 
    int lvid ;

    int typecode ; 
    int param ; 
    int aabb ; 
    int xform ; 




    void init(); 
    std::string brief() const ; 
    std::string desc() const ; 

    void setTypecode( int tc ); 
    void setParam( double x,  double y,  double z,  double w,  double z1, double z2 ); 
    void setAABB(  double x0, double y0, double z0, double x1, double y1, double z1 );
    void setXForm( const glm::tmat4x4<double>& t ); 

    void check_z() const ; 
    double zmin() const ; 
    double zmax() const ; 
    void decrease_zmin( double dz ); 
    void increase_zmax( double dz ); 
    static std::string ZDesc(const std::vector<int>& prims);
    static void ZNudgeEnds(    const std::vector<int>& prims); 
    static void ZNudgeJoints(  const std::vector<int>& prims); 

    // signatures need to match CSGNode where these param will end up 
    static int Boolean( int op, int l, int r ); 
    static int Compound(int type, const std::vector<int>& prims ); 
    static int Cylinder(double radius, double z1, double z2) ;
    static int Cone(double r1, double z1, double r2, double z2); 
    static int Sphere(double radius); 
    static int ZSphere(double radius, double z1, double z2); 
    static int Box3(double fullside); 
    static int Box3(double fx, double fy, double fz ); 
    static int Zero(double  x,  double y,  double z,  double w,  double z1, double z2); 
};


