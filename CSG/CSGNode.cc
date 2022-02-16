#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector_types.h>

#include "scuda.h"
#include "SThetaCut.hh"
#include "SPhiCut.hh"
#include "OpticksCSG.h"
#include "PLOG.hh"

#include "CSGNode.h"


const float CSGNode::UNBOUNDED_DEFAULT_EXTENT = 100.f ; 


std::string CSGNode::Addr(unsigned repeatIdx, unsigned primIdx, unsigned partIdxRel ) // static
{
    int wid = 10 ; 
    std::stringstream ss ; 
    ss  
        << std::setw(2) << std::setfill('0') << repeatIdx 
        << "/" 
        << std::setw(3) << std::setfill('0') << primIdx 
        << "/" 
        << std::setw(3) << std::setfill('0') << partIdxRel
        ; 
    std::string s = ss.str(); 
    std::stringstream tt ; 
    tt << std::setw(wid) << s ; 
    std::string t = tt.str(); 
    return t ; 
}

std::string CSGNode::Desc(const float* fval, int numval, int wid, int prec ) // static 
{
    std::stringstream ss ; 
    for(int p=0 ; p < numval ; p++) 
        ss << std::fixed << std::setw(wid) << std::setprecision(prec) << *(fval+p) << " "  ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string CSGNode::desc() const 
{
    const float* aabb = AABB(); 
    unsigned trIdx = gtransformIdx(); 

    std::stringstream ss ; 
    ss
        << "CSGNode "
        << std::setw(5) << index() 
        << " " 
        << brief()
        << " aabb: " << Desc( aabb, 6, 7, 1 ) 
        << " trIdx: " << std::setw(5) << trIdx 
        ;    

    std::string s = ss.str(); 
    return s ; 
}

std::string CSGNode::tag() const
{
    return CSG::Tag((OpticksCSG_t)typecode()) ; 
}

std::string CSGNode::brief() const
{
    std::stringstream ss ; 
    ss
        << ( is_complement() ? "!" : " " )
        << tag()
        ;
    std::string s = ss.str(); 
    return s ; 
}



void CSGNode::Dump(const CSGNode* n_, unsigned ni, const char* label)
{
    std::cout << "CSGNode::Dump ni " << ni << " " ; 
    if(label) std::cout << label ;  
    std::cout << std::endl ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const CSGNode* n = n_ + i ;        
        std::cout << "(" << i << ")" << std::endl ; 
        std::cout 
            << " node.q0.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.w
            << " ) " 
            << std::endl 
            << " node.q1.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.w
            << " ) " 
            << std::endl 
            << " node.q2.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.w
            << " ) " 
            << std::endl 
            << " node.q3.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.w
            << " ) " 
            << std::endl 
            ;

        std::cout 
            << " node.q0.i.xyzw ( " 
            << std::setw(10) << n->q0.i.x  
            << std::setw(10) << n->q0.i.y  
            << std::setw(10) << n->q0.i.z  
            << std::setw(10) << n->q0.i.w
            << " ) " 
            << std::endl 
            << " node.q1.i.xyzw ( " 
            << std::setw(10) << n->q1.i.x  
            << std::setw(10) << n->q1.i.y  
            << std::setw(10) << n->q1.i.z  
            << std::setw(10) << n->q1.i.w
            << " ) " 
            << std::endl 
            << " node.q2.i.xyzw ( " 
            << std::setw(10) << n->q2.i.x  
            << std::setw(10) << n->q2.i.y  
            << std::setw(10) << n->q2.i.z  
            << std::setw(10) << n->q2.i.w
            << " ) " 
            << std::endl 
            << " node.q3.i.xyzw ( " 
            << std::setw(10) << n->q3.i.x  
            << std::setw(10) << n->q3.i.y  
            << std::setw(10) << n->q3.i.z  
            << std::setw(10) << n->q3.i.w
            << " ) " 
            << std::endl 
            ;
    }
}



CSGNode CSGNode::Union(){         return CSGNode::BooleanOperator('U') ; }  // static
CSGNode CSGNode::Intersection(){  return CSGNode::BooleanOperator('I') ; }
CSGNode CSGNode::Difference(){    return CSGNode::BooleanOperator('D') ; }
CSGNode CSGNode::BooleanOperator(char op)   // static 
{
    CSGNode nd = {} ;
    nd.setTypecode(CSG::BooleanOperator(op)) ; 
    return nd ; 
}

CSGNode CSGNode::Overlap(   unsigned num_sub){    return CSGNode::ListHeader( 'O', num_sub ); }
CSGNode CSGNode::Contiguous(unsigned num_sub){    return CSGNode::ListHeader( 'C', num_sub ); }
CSGNode CSGNode::Discontiguous(unsigned num_sub){ return CSGNode::ListHeader( 'D', num_sub ); }
CSGNode CSGNode::ListHeader(char type, unsigned num_sub)   // static 
{
    CSGNode nd = {} ;
    switch(type)
    { 
        case 'O': nd.setTypecode(CSG_OVERLAP)      ; break ; 
        case 'C': nd.setTypecode(CSG_CONTIGUOUS) ; break ; 
        case 'D': nd.setTypecode(CSG_DISCONTIGUOUS) ; break ; 
    }
    nd.setSubNum(num_sub); 
    return nd ; 
}



bool CSGNode::is_operator() const 
{
    unsigned tc = typecode(); 
    return CSG::IsOperator((OpticksCSG_t)tc) ;
}
bool CSGNode::is_intersection() const 
{
    unsigned tc = typecode(); 
    return CSG::IsIntersection((OpticksCSG_t)tc) ;
}
bool CSGNode::is_union() const 
{
    unsigned tc = typecode(); 
    return CSG::IsUnion((OpticksCSG_t)tc) ;
}
bool CSGNode::is_difference() const 
{
    unsigned tc = typecode(); 
    return CSG::IsDifference((OpticksCSG_t)tc) ;
}
bool CSGNode::is_zero() const 
{
    unsigned tc = typecode(); 
    return CSG::IsZero((OpticksCSG_t)tc)  ;
}
bool CSGNode::is_primitive() const 
{
    unsigned tc = typecode(); 
    return CSG::IsPrimitive((OpticksCSG_t)tc)  ;
}
bool CSGNode::is_complemented_primitive() const 
{
    return is_complement() && is_primitive() ; 
}







/**
CSGNode::AncestorTypeMask
---------------------------

*partIdxRel* is the zero based node index, zero corresponding to root

This iterates up the node tree starting from the parent 
of the node identified by *partIdxRel* using complete binary tree arithemetic.
The bitwise-or of the typemasks of the ancestors is returned.  
When starting from a leaf node this will return the bitwise-or of its 
ancestors operator types. 

Note that as this only goes upwards in the tree it can be used
during tree creation/convertion, eg from CSG_GGeo_Convert::convertPrim.
Of course the CSGNode identified by *partIdxRel* MUST already exist.

**/

unsigned CSGNode::AncestorTypeMask( const CSGNode* root,  unsigned partIdxRel, bool dump  ) // static
{
    unsigned atm = 0u ; 

    int parentIdxRel = ((partIdxRel + 1) >> 1) - 1 ;   // make 1-based, bitshift up tree, return to 0-based   

    while( parentIdxRel > -1 )  
    {   
        const CSGNode* p = root + parentIdxRel ;  

        atm |= p->typemask() ;  // NB typemask NOT typecode to allow meaningful bitwise-ORing  

        if(dump) std::cout << std::setw(3) << parentIdxRel << " " << p->desc() << std::endl ;
                
        parentIdxRel = ((parentIdxRel + 1) >> 1) - 1 ;  
    }   

    return atm ; 
}


/**
CSGNode::Depth
----------------

Return complete binary tree depth from 0-based node index (aka *partIdxRel*) see CSGNodeTest/test_Depth::

     partIdxRel    0 partIdxRel+1 (dec)    1 (bin) 00000000000000000000000000000001 depth    0
     partIdxRel    1 partIdxRel+1 (dec)    2 (bin) 00000000000000000000000000000010 depth    1
     partIdxRel    2 partIdxRel+1 (dec)    3 (bin) 00000000000000000000000000000011 depth    1
     partIdxRel    3 partIdxRel+1 (dec)    4 (bin) 00000000000000000000000000000100 depth    2
     partIdxRel    4 partIdxRel+1 (dec)    5 (bin) 00000000000000000000000000000101 depth    2
     partIdxRel    5 partIdxRel+1 (dec)    6 (bin) 00000000000000000000000000000110 depth    2
     partIdxRel    6 partIdxRel+1 (dec)    7 (bin) 00000000000000000000000000000111 depth    2
     partIdxRel    7 partIdxRel+1 (dec)    8 (bin) 00000000000000000000000000001000 depth    3
     partIdxRel    8 partIdxRel+1 (dec)    9 (bin) 00000000000000000000000000001001 depth    3
     partIdxRel    9 partIdxRel+1 (dec)   10 (bin) 00000000000000000000000000001010 depth    3
     partIdxRel   10 partIdxRel+1 (dec)   11 (bin) 00000000000000000000000000001011 depth    3
     partIdxRel   11 partIdxRel+1 (dec)   12 (bin) 00000000000000000000000000001100 depth    3
     partIdxRel   12 partIdxRel+1 (dec)   13 (bin) 00000000000000000000000000001101 depth    3
     partIdxRel   13 partIdxRel+1 (dec)   14 (bin) 00000000000000000000000000001110 depth    3
     partIdxRel   14 partIdxRel+1 (dec)   15 (bin) 00000000000000000000000000001111 depth    3
     partIdxRel   15 partIdxRel+1 (dec)   16 (bin) 00000000000000000000000000010000 depth    4
     partIdxRel   16 partIdxRel+1 (dec)   17 (bin) 00000000000000000000000000010001 depth    4
     partIdxRel   17 partIdxRel+1 (dec)   18 (bin) 00000000000000000000000000010010 depth    4

**/

unsigned CSGNode::Depth( unsigned partIdxRel ) // static
{
    int parentIdxRel = ((partIdxRel + 1) >> 1) - 1 ;   // make 1-based, bitshift up tree, return to 0-based   
    unsigned depth = 0 ; 
    while( parentIdxRel > -1 )  
    {   
        parentIdxRel = ((parentIdxRel + 1) >> 1) - 1 ;  
        depth += 1 ; 
    }   
    return depth ; 
}




bool CSGNode::IsOnlyUnionMask( unsigned atm )  // static 
{
     return atm == CSG::UnionMask() ; 
}

bool CSGNode::IsOnlyIntersectionMask( unsigned atm )  // static 
{
     return atm == CSG::IntersectionMask() ; 
}

bool CSGNode::IsOnlyDifferenceMask( unsigned atm )  // static 
{
     return atm == CSG::DifferenceMask() ; 
}

void CSGNode::getYRange(float& y0, float& y1) const 
{
    unsigned tc = typecode(); 
    if( tc == CSG_BOX3 )
    {
        float fx, fy, fz, a, b, c ; 
        getParam( fx, fy, fz, a, b, c ); 
  
        y0 = -fy*0.5f ; 
        y1 =  fy*0.5f ;  
    }
}

void CSGNode::setAABBLocal()
{
    unsigned tc = typecode(); 
    if(tc == CSG_SPHERE)
    {
        float r = radius();  
        setAABB(  -r, -r, -r,  r, r, r  ); 
    }
    else if(tc == CSG_ZSPHERE)
    {
        float r = radius();  
        float z1_ = z1();  
        float z2_ = z2();  
        assert( z2_ > z1_ ); 
        setAABB(  -r, -r, z1_,  r, r, z2_  ); 
    }
    else if( tc == CSG_CONE )
    {
        float r1, z1, r2, z2, a, b ; 
        getParam( r1, z1, r2, z2, a, b ); 
        float rmax = fmaxf(r1, r2) ;
        setAABB( -rmax, -rmax, z1, rmax, rmax, z2 ); 
    }
    else if( tc == CSG_BOX3 )
    {
        float fx, fy, fz, a, b, c ; 
        getParam( fx, fy, fz, a, b, c ); 
        setAABB( -fx*0.5f , -fy*0.5f, -fz*0.5f, fx*0.5f , fy*0.5f, fz*0.5f );   
    }
    else if( tc == CSG_CYLINDER )
    {
        float px, py, a, radius, z1, z2 ; 
        getParam( px, py, a, radius, z1, z2) ; 
        setAABB( px-radius, py-radius, z1, px+radius, py+radius, z2 );   
    }
    else if( tc == CSG_DISC )
    {
        float px, py, ir, r, z1, z2 ; 
        getParam( px, py, ir, r, z1, z2 ); 
        setAABB( px - r , py - r , z1, px + r, py + r, z2 ); 
    }
    else if( tc == CSG_HYPERBOLOID )
    {
        float r0, zf, z1, z2, a, b ; 
        getParam(r0, zf, z1, z2, a, b ) ; 

        assert( z1 < z2 ); 
        const float rr0 = r0*r0 ; 
        const float z1s = z1/zf ; 
        const float z2s = z2/zf ; 

        const float rr1 = rr0 * ( z1s*z1s + 1.f ) ;
        const float rr2 = rr0 * ( z2s*z2s + 1.f ) ;
        const float rmx = sqrtf(fmaxf( rr1, rr2 )) ; 

        setAABB(  -rmx,  -rmx,  z1,  rmx, rmx, z2 ); 
    }
    else if( tc == CSG_UNION || tc == CSG_INTERSECTION || tc == CSG_DIFFERENCE )
    {
        setAABB( 0.f );  
    }
    else if( tc == CSG_CONTIGUOUS || tc == CSG_DISCONTIGUOUS )
    {
        setAABB( 0.f );  
    }
    else if( tc == CSG_ZERO )
    {
        setAABB( UNBOUNDED_DEFAULT_EXTENT ); 
    }
    else if( tc == CSG_PHICUT || tc == CSG_LPHICUT )
    {
        setAABB( UNBOUNDED_DEFAULT_EXTENT ); 
    }
    else if( tc == CSG_THETACUT || tc == CSG_LTHETACUT )
    {
        setAABB( UNBOUNDED_DEFAULT_EXTENT ); 
    }
    else
    {
        LOG(fatal) << " not implemented for tc " << tc << " CSG::Name(tc) " << CSG::Name(tc) ; 
        assert(0); 
        setAABB( 0.f );  
    }

}

CSGNode CSGNode::Zero()  // static
{
    CSGNode nd = {} ;
    nd.setAABB(  -0.f, -0.f, -0.f,  0.f, 0.f, 0.f  ); 
    nd.setTypecode(CSG_ZERO) ; 
    return nd ; 
}
CSGNode CSGNode::Sphere(float radius)  // static
{
    assert( radius > 0.f); 
    CSGNode nd = {} ;
    nd.setParam( 0.f, 0.f, 0.f, radius,  0.f,  0.f ); 
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  ); 
    nd.setTypecode(CSG_SPHERE) ; 
    return nd ;
}
CSGNode CSGNode::ZSphere(float radius, float z1, float z2)  // static
{
    assert( radius > 0.f); 
    assert( z2 > z1 ); 
    CSGNode nd = {} ;
    nd.setParam( 0.f, 0.f, 0.f, radius, z1, z2 ); 
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  ); 
    nd.setTypecode(CSG_ZSPHERE) ; 
    return nd ;
}
CSGNode CSGNode::Cone(float r1, float z1, float r2, float z2)  // static
{
    assert( z2 > z1 ); 
    float rmax = fmaxf(r1, r2) ;
    CSGNode nd = {} ;
    nd.setParam( r1, z1, r2, z2, 0.f, 0.f ) ;
    nd.setAABB( -rmax, -rmax, z1, rmax, rmax, z2 ); 
    nd.setTypecode(CSG_CONE) ; 
    return nd ; 
}
CSGNode CSGNode::Box3(float fullside)  // static 
{
    return Box3(fullside, fullside, fullside); 
}
CSGNode CSGNode::Box3(float fx, float fy, float fz )  // static 
{
    assert( fx > 0.f ); 
    assert( fy > 0.f ); 
    assert( fz > 0.f ); 

    CSGNode nd = {} ;
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f ); 
    nd.setAABB( -fx*0.5f , -fy*0.5f, -fz*0.5f, fx*0.5f , fy*0.5f, fz*0.5f );   
    nd.setTypecode(CSG_BOX3) ; 
    return nd ; 
}
CSGNode CSGNode::Cylinder(float px, float py, float radius, float z1, float z2)
{
    CSGNode nd = {} ; 
    nd.setParam( px, py, 0.f, radius, z1, z2)  ; 
    nd.setAABB( px-radius, py-radius, z1, px+radius, py+radius, z2 );   
    nd.setTypecode(CSG_CYLINDER); 
    return nd ; 
} 

CSGNode CSGNode::InfCylinder(float radius, float hz)
{
    assert( hz > 0.f );  // need to bound it ? 
    CSGNode nd = {} ; 
    nd.setParam( 0.f, 0.f, 0.f, radius, 0.f,0.f)  ; 
    nd.setAABB( -radius, -radius, -hz,  radius, radius, hz );   
    nd.setTypecode(CSG_INFCYLINDER); 
    return nd ; 
} 

CSGNode CSGNode::InfPhiCut(float startPhi_pi, float deltaPhi_pi ) 
{
    CSGNode nd = {} ; 
    SPhiCut::PrepareParam( nd.q0 ,  startPhi_pi, deltaPhi_pi ); 
    nd.setAABB( -100.f,-100.f,-100.f, 100.f, 100.f, 100.f );      // placeholder, hmm how to avoid ?
    nd.setTypecode(CSG_PHICUT); 
    return nd ; 
}

CSGNode CSGNode::InfThetaCut(float startTheta_pi, float deltaTheta_pi, char imp) 
{
    CSGNode nd = {} ; 
    SThetaCut::PrepareParam( nd.q0, nd.q1, startTheta_pi, deltaTheta_pi ); 
    nd.setAABB( -100.f,-100.f,-100.f, 100.f, 100.f, 100.f );    // HMM: adhoc ?   
    nd.setTypecode( imp == 'L' ? CSG_LTHETACUT : CSG_THETACUT ); 
    return nd ; 
}


CSGNode CSGNode::Disc(float px, float py, float ir, float r, float z1, float z2)
{
    CSGNode nd = {} ;
    nd.setParam( px, py, ir, r, z1, z2 ); 
    nd.setAABB( px - r , py - r , z1, px + r, py + r, z2 ); 
    nd.setTypecode(CSG_DISC); 
    return nd ; 
} 

CSGNode CSGNode::Hyperboloid(float r0, float zf, float z1, float z2) // static
{
    assert( z1 < z2 ); 
    const float rr0 = r0*r0 ; 
    const float z1s = z1/zf ; 
    const float z2s = z2/zf ; 

    const float rr1 = rr0 * ( z1s*z1s + 1.f ) ;
    const float rr2 = rr0 * ( z2s*z2s + 1.f ) ;
    const float rmx = sqrtf(fmaxf( rr1, rr2 )) ; 

    CSGNode nd = {} ;
    nd.setParam(r0, zf, z1, z2, 0.f, 0.f ) ; 
    nd.setAABB(  -rmx,  -rmx,  z1,  rmx, rmx, z2 ); 
    nd.setTypecode(CSG_HYPERBOLOID) ; 
    return nd ; 
}


CSGNode CSGNode::Plane(float nx, float ny, float nz, float d)
{
    CSGNode nd = {} ;
    nd.setParam(nx, ny, nz, d, 0.f, 0.f ) ;
    nd.setTypecode(CSG_PLANE) ; 
    nd.setAABB( UNBOUNDED_DEFAULT_EXTENT ); 
    return nd ; 
}

CSGNode CSGNode::Slab(float nx, float ny, float nz, float d1, float d2 )
{
    CSGNode nd = {} ;
    nd.setParam( nx, ny, nz, 0.f, d1, d2 ); 
    nd.setTypecode(CSG_SLAB) ; 
    nd.setAABB( UNBOUNDED_DEFAULT_EXTENT ); 
    return nd ; 
}



/**
CSGNode::MakeDemo
-------------------

Only the first four chars of the name are used to select the type of node.

* see CSGMaker for more extensive test node/prim/solid creation

**/

CSGNode CSGNode::MakeDemo(const char* name) // static
{
    if(strncmp(name, "sphe", 4) == 0) return CSGNode::Sphere(100.f) ; 
    if(strncmp(name, "zsph", 4) == 0) return CSGNode::ZSphere(100.f, -50.f, 50.f) ; 
    if(strncmp(name, "cone", 4) == 0) return CSGNode::Cone(150.f, -150.f, 50.f, -50.f) ; 
    if(strncmp(name, "hype", 4) == 0) return CSGNode::Hyperboloid(100.f, 50.f, -50.f, 50.f) ; 
    if(strncmp(name, "box3", 4) == 0) return CSGNode::Box3(100.f, 100.f, 100.f) ; 
    if(strncmp(name, "plan", 4) == 0) return CSGNode::Plane(1.f, 0.f, 0.f, 0.f) ; 
    if(strncmp(name, "slab", 4) == 0) return CSGNode::Slab(1.f, 0.f, 0.f, -10.f, 10.f ) ; 
    if(strncmp(name, "cyli", 4) == 0) return CSGNode::Cylinder(0.f, 0.f, 100.f, -50.f, 50.f ) ; 
    if(strncmp(name, "disc", 4) == 0) return CSGNode::Disc(    0.f, 0.f, 50.f, 100.f, -2.f, 2.f ) ; 
    if(strncmp(name, "iphi", 4) == 0) return CSGNode::InfPhiCut(0.25f, 0.10f) ; 
    if(strncmp(name, "ithe", 4) == 0) return CSGNode::InfThetaCut(0.25f, 0.10f, ' ') ; 
    if(strncmp(name, "ithl", 4) == 0) return CSGNode::InfThetaCut(0.25f, 0.10f, 'L') ; 
    LOG(fatal) << " not implemented for name " << name ; 
    assert(0); 
    return CSGNode::Sphere(1.0); 
}

/**
CSGNode::Make
--------------

NB not easy to expand to more than 6 params as q1.u.z q1.u.w are otherwise engaged 

**/

CSGNode CSGNode::Make(unsigned typecode, const float* param6, const float* aabb ) // static
{
    CSGNode nd = {} ;
    nd.setTypecode(typecode) ; 
    if(param6) nd.setParam( param6 );  
    if(aabb)    nd.setAABB( aabb );  
    return nd ; 
}







#endif

