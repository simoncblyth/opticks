#pragma once

struct CSGNode ; 
struct CSGSolid ; 

#include <string>
#include <vector>
#include "CSG_API_EXPORT.hh"
#include "plog/Severity.h"

struct CSGFoundry ; 

struct CSG_API CSGMaker
{
    static const plog::Severity LEVEL ; 

    static const char* NAMES ; 
    static bool StartsWith( const char* n, const char* q ); 
    static bool CanMake( const char* q ); 
    static void GetNames(std::vector<std::string>& names ); 

    static float4 TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k );

    enum {  // enum used for Demo solids (equivalent to lvIdx or meshIdx with full geometries)
        SPHE_MIDX, ZSPH_MIDX, CONE_MIDX, HYPE_MIDX, BOX3_MIDX, 
        PLAN_MIDX, SLAB_MIDX, CYLI_MIDX, DISC_MIDX, VCUB_MIDX, 
        VTET_MIDX, ELLI_MIDX, UBSP_MIDX, IBSP_MIDX, DBSP_MIDX, 
        RCYL_MIDX, ICYL_MIDX, IPHI_MIDX, ITHE_MIDX, ITHL_MIDX
    }; 

    CSGFoundry* fd ; 
    CSGMaker( CSGFoundry* fd );  

    void makeDemoGrid();
    void makeDemoSolids() ;

    CSGSolid* make(const char* name); 

    CSGSolid* makeLayered( const char* label, float outer_radius, unsigned layers ) ;
    CSGSolid* makeScaled(const char* label, const char* demo_node_type, float outer_scale, unsigned layers );
    CSGSolid* makeClustered(const char* name,  int i0, int i1, int is, int j0, int j1, int js, int k0, int k1, int ks, double unit, bool inbox ) ;

    CSGSolid* makeSolid11(const char* label, CSGNode nd, const std::vector<float4>* pl=nullptr, int meshIdx=-1 );
    CSGSolid* makeBooleanBoxSphere( const char* label, char op, float radius, float fullside, int meshIdx = -1  ) ;
    CSGSolid* makeBooleanTriplet(   const char* label, char op, const CSGNode& left, const CSGNode& right, int meshIdx=-1 ) ; 

    CSGSolid* makeOverlapList(       const char* label,            const std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* trans  ); 
    CSGSolid* makeContiguousList(    const char* label,            const std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* trans  ); 
    CSGSolid* makeDiscontiguousList( const char* label,            const std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* trans  ); 
    CSGSolid* makeList(              const char* label, char type, const std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* trans  );
    CSGSolid* makeListOne(           const char* label, char type, const CSGNode& leaf ); 


    CSGSolid* makeBooleanSeptuplet( 
        const char* label, 
        const CSGNode& top, 
        const CSGNode& l, 
        const CSGNode& r, 
        const CSGNode& ll, 
        const CSGNode& lr, 
        const CSGNode& rl, 
        const CSGNode& rr, 
        const int meshIdx 
    ); 

    CSGSolid* makeDifferenceCylinder( const char* label="dcyl", float rmax=100.f, float rmin=50.f, float z1=-50.f, float z2=50.f, float z_inner_factor=1.01f   ); 
    CSGSolid* makeBoxSubSubCylinder(  const char* label="bssc", float fullside=300.f, float rmax=100.f, float rmin=50.f, float z1=-50.f, float z2=50.f, float z_inner_factor=1.01 );

    CSGSolid* makeUnionBoxSphere(        const char* label="ubsp", float radius=100.f, float fullside=150.f );
    CSGSolid* makeUnionListBoxSphere(    const char* label="UnionListBoxSphere", float radius=100.f, float fullside=150.f );
    CSGSolid* makeIntersectionBoxSphere( const char* label="ibsp", float radius=100.f, float fullside=150.f );
    CSGSolid* makeOverlapBoxSphere(       const char* label="obsp", float radius=100.f, float fullside=150.f ); 


    CSGSolid* makeListThreeSphere(          const char* label, char type, float radius, float side  ); 
    CSGSolid* makeOverlapThreeSphere(       const char* label, float radius=100.f, float side=50.f ); 
    CSGSolid* makeContiguousThreeSphere(    const char* label, float radius=100.f, float side=50.f ); 
    CSGSolid* makeDiscontiguousThreeSphere( const char* label, float radius=100.f, float side=100.f  ); 


    CSGSolid* makeContiguousBoxSphere(    const char* label="cbsp", float radius=100.f, float fullside=150.f ); 
    CSGSolid* makeDiscontiguousBoxSphere( const char* label="DiscontiguousBoxSphere", float radius=100.f, float fullside=150.f ) ; 

    CSGSolid* makeDifferenceBoxSphere(   const char* label="dbsp", float radius=100.f, float fullside=150.f );

    CSGSolid* makeSphere(     const char* label="sphe", float r=100.f ); 
    CSGSolid* makeEllipsoid(  const char* label="elli", float rx=100.f, float ry=100.f, float rz=50.f ); 

    CSGSolid* makeRotatedCylinder(const char* label="rcyl", float px=0.f, float py=0.f, float radius=100.f, float z1=-50.f, float z2=50.f, float ax=1.f, float ay=0.f, float az=0.f, float angle_deg=45.f  );

    CSGSolid* makeInfCylinder(const char* label="icyl", float radius=50.f,  float hz=25.f ); 
    CSGSolid* makeInfPhiCut(  const char* label="iphi", float startPhi_pi=0.25f,   float deltaPhi_pi=0.1f ); // units of pi  
    CSGSolid* makeInfThetaCut(  const char* label="ithe", float startTheta_pi=0.25f, float deltaTheta_pi=0.1f ); // units of pi  
    CSGSolid* makeInfThetaCutL( const char* label="ithl", float startTheta_pi=0.25f, float deltaTheta_pi=0.1f ); // units of pi  


    CSGSolid* makeZSphere(    const char* label="zsph", float r=100.f,  float z1=-50.f , float z2=50.f ); 
    CSGSolid* makeCone(       const char* label="cone", float r1=300.f, float z1=-300.f, float r2=100.f,   float z2=-100.f ); 
    CSGSolid* makeHyperboloid(const char* label="hype", float r0=100.f, float zf=50.f,   float z1=-50.f,   float z2=50.f );
    CSGSolid* makeBox3(       const char* label="box3", float fx=100.f, float fy=200.f,  float fz=300.f );
    CSGSolid* makePlane(      const char* label="plan", float nx=1.0f,  float ny=0.f,    float nz=0.f,     float d=0.f );
    CSGSolid* makeSlab(       const char* label="slab", float nx=1.0f,  float ny=0.f,    float nz=0.f,     float d1=-50.f, float d2=50.f );
    CSGSolid* makeCylinder(   const char* label="cyli", float px=0.f,   float py=0.f,    float r=100.f,    float z1=-50.f, float z2=50.f );
    CSGSolid* makeDisc(       const char* label="disc", float px=0.f,   float py=0.f,    float ir=50.f,    float r=100.f,  float z1=-2.f, float z2=2.f);

    CSGSolid* makeConvexPolyhedronCube(       const char* label="vcub", float extent=100.f );
    CSGSolid* makeConvexPolyhedronTetrahedron(const char* label="vtet", float extent=100.f);



};


