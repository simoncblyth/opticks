#pragma once
#include "squad.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define NODE_METHOD __device__
#else
   #define NODE_METHOD 
#endif 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#endif

#include "CSG_API_EXPORT.hh"


/**
CSGNode (synonymous with Part)
==============================

NB elements are used for different purposes depending on typecode, 
eg planeIdx, planeNum are used only with CSG_CONVEXPOLYHEDRON.  Marked "cx:" below.


sp:sphere
   center, radius 

zs:zsphere
   center, radius, z1, z2 cuts 

cy:cylinder
   center, radius, z1, z2

ds:disc
   very flat cylinder

cn:cone
   r1, z1, r2, z2 

hy:hyperboloid
   r0 (z=0 waist), ...

pl:plane (unbounded)
   normal and distance from origin 

sl:slab (unbounded)
   normal and two distances from origin 

cx:convexpolyhedron 
   planeOffset and number of planes   



* vim replace : shift-R


    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    | q  |      x         |      y         |     z          |      w         |  notes                                          |
    +====+================+================+================+================+=================================================+
    |    | sp/zs/cy:cen_x | sp/zs/cy:cen_y | sp/zs/cy:cen_z | sp/zs/cy:radius|  eliminate center? as can be done by transform  |
    | q0 | cn:r1          | cn:z1          | cn:r2          | cn:z2          |  cn:z2 > z1                                     |
    |    | hy:r0 z=0 waist| hy:zf          | hy:z1          | hy:z2          |  hy:z2 > z1                                     |
    |    | b3:fx          | b3:fy          | b3:fz          |                |  b3: fullside dimensions, center always origin  |
    |    | pl/sl:nx       | pl/sl:ny       | pl/sl:nz       | pl:d           |  pl: NB Node plane distinct from plane array    |
    |    |                |                | ds:inner_r     | ds:radius      |                                                 |
    |    |                |                |                | radius()       |                                                 |
    |    | cx:planeIdx    | cx:planeNum    |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    | zs:zdelta_0    | zs:zdelta_1    | boundary       | index          |                                                 |
    |    | sl:a           | sl:b           |                |                |  sl:a,b offsets from origin                     |
    | q1 | cy:z1          | cy:z2          |                |                |  cy:z2 > z1                                     |
    |    | ds:z1          | ds:z2          |                |                |                                                 |
    |    | z1()           | z2()           |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |                |                |  q2.w was previously typecode                   |
    |    |                |                |                |                |                                                 |
    | q2 |  BBMin_x       |  BBMin_y       |  BBMin_z       |  BBMax_x       |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |  typecode      | gtransformIdx  |                                                 |
    |    |                |                |                | complement     |                                                 |
    | q3 |  BBMax_y       |  BBMax_z       |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+


* moved typecode from q2.w in order to give 6 contiguous slots for aabb

**/

struct CSG_API CSGNode
{
    quad q0 ;
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    NODE_METHOD unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
    NODE_METHOD unsigned planeNum()      const { return q0.u.y ; } 
    NODE_METHOD void setPlaneIdx(unsigned idx){  q0.u.x = idx ; } 
    NODE_METHOD void setPlaneNum(unsigned num){  q0.u.y = num ; }

    NODE_METHOD void getParam( float& x , float& y , float& z , float& w , float& z1, float& z2 )
    {
        x = q0.f.x ; 
        y = q0.f.y ; 
        z = q0.f.z ; 
        w = q0.f.w ; 
        z1 = q1.f.x ;
        z2 = q1.f.y ;  
    }
    NODE_METHOD void setParam( float  x , float  y , float  z , float  w , float  z1, float  z2 )
    { 
        q0.f.x = x  ; 
        q0.f.y = y  ; 
        q0.f.z = z  ; 
        q0.f.w = w  ; 
        q1.f.x = z1 ; 
        q1.f.y = z2 ;  
    }

    NODE_METHOD void setParam(const float* p)
    { 
        q0.f.x = *(p+0) ; 
        q0.f.y = *(p+1) ; 
        q0.f.z = *(p+2) ; 
        q0.f.w = *(p+3) ; 
        q1.f.x = *(p+4) ; 
        q1.f.y = *(p+5) ;  
    }

    NODE_METHOD void setAABBLocal();  // sets local frame BBox based on typecode and parameters (WARNING: not implemented for all shapes yet)
    NODE_METHOD void setAABB(  float x0, float y0, float z0, float x1, float y1, float z1){  q2.f.x = x0 ; q2.f.y = y0 ; q2.f.z = z0 ; q2.f.w = x1 ; q3.f.x = y1 ; q3.f.y = z1 ; }  
    NODE_METHOD void setAABB(  float e ){                                                    q2.f.x = -e ; q2.f.y = -e ; q2.f.z = -e ; q2.f.w =  e ; q3.f.x =  e ; q3.f.y =  e ; }  
    NODE_METHOD void setAABB(const float* p)
    { 
        q2.f.x = *(p+0) ; 
        q2.f.y = *(p+1) ; 
        q2.f.z = *(p+2) ; 
        q2.f.w = *(p+3) ; 
        q3.f.x = *(p+4) ; 
        q3.f.y = *(p+5) ;  
    }


    NODE_METHOD       float* AABB()       {  return &q2.f.x ; }
    NODE_METHOD const float* AABB() const {  return &q2.f.x ; }
    NODE_METHOD const float3 mn() const {    return make_float3(q2.f.x, q2.f.y, q2.f.z) ; }
    NODE_METHOD const float3 mx() const {    return make_float3(q2.f.w, q3.f.x, q3.f.y) ; }
    NODE_METHOD float extent() const 
    {
        float3 d = make_float3( q2.f.w - q2.f.x, q3.f.x - q2.f.y, q3.f.y - q2.f.z ); 
        return fmaxf(fmaxf(d.x, d.y), d.z) /2.f ; 
    }


    NODE_METHOD unsigned boundary()  const {      return q1.u.z ; }   
    NODE_METHOD void setBoundary(unsigned bnd){          q1.u.z = bnd ; }

    NODE_METHOD unsigned index()     const {      return q1.u.w ; }    
    NODE_METHOD void setIndex(unsigned idx){             q1.u.w = idx ; }

    NODE_METHOD unsigned typecode()  const {      return q3.u.z ; }  //  OptickCSG_t enum 
    NODE_METHOD void setTypecode(unsigned tc){           q3.u.z = tc ; }

    NODE_METHOD unsigned typemask()  const {      return 1 << q3.u.z ; } //  mask integer suitable for bitwise-oring  

    NODE_METHOD void zeroTransformComplement(){         q3.u.w = 0 ; }  
    NODE_METHOD void setTransform(  unsigned idx ){     q3.u.w |= (idx & 0x7fffffff) ; }
    NODE_METHOD void setComplement( bool complement ){  q3.u.w |= ( (int(complement) << 31) & 0x80000000) ; }



    NODE_METHOD unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None 
    NODE_METHOD bool     is_complement() const { return q3.u.w & 0x80000000 ; } 


    NODE_METHOD float radius() const { return q0.f.w ; } ;
    NODE_METHOD float z1() const {     return q1.f.x ; } ;
    NODE_METHOD float z2() const {     return q1.f.y ; } ;



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

    static std::string Addr(unsigned repeatIdx, unsigned primIdx, unsigned partIdxRel ); 

    static std::string Desc(const float* fval, int numval=6, int wid=7, int prec=1 ); 
    std::string desc() const ; 
    static void Dump(const CSGNode* n, unsigned ni, const char* label);  

    bool is_operator() const ; 
    bool is_intersection() const ; 
    bool is_union() const ; 
    bool is_difference() const ; 
    bool is_primitive() const ; 
    bool is_complemented_primitive() const ; 
    bool is_zero() const ; 

    static unsigned AncestorTypeMask( const CSGNode* root, unsigned partIdxRel, bool dump  ); 
    static unsigned Depth( unsigned partIdxRel ); 
    static bool     IsOnlyUnionMask( unsigned atm ); 
    static bool     IsOnlyIntersectionMask( unsigned atm ); 
    static bool     IsOnlyDifferenceMask( unsigned atm ); 

    static void Copy(CSGNode& b, const CSGNode& a)
    {
        b.q0.f.x = a.q0.f.x ; b.q0.f.y = a.q0.f.y ; b.q0.f.z = a.q0.f.z ; b.q0.f.w = a.q0.f.w ; 
        b.q1.f.x = a.q1.f.x ; b.q1.f.y = a.q1.f.y ; b.q1.f.z = a.q1.f.z ; b.q1.f.w = a.q1.f.w ; 
        b.q2.f.x = a.q2.f.x ; b.q2.f.y = a.q2.f.y ; b.q2.f.z = a.q2.f.z ; b.q2.f.w = a.q2.f.w ; 
        b.q3.f.x = a.q3.f.x ; b.q3.f.y = a.q3.f.y ; b.q3.f.z = a.q3.f.z ; b.q3.f.w = a.q3.f.w ; 
    }

    static const float UNBOUNDED_DEFAULT_EXTENT ; 

    static CSGNode Union(); 
    static CSGNode Intersection(); 
    static CSGNode Difference(); 
    static CSGNode BooleanOperator(char op); 

    static CSGNode Zero();
    static CSGNode Sphere(float radius);
    static CSGNode ZSphere(float radius, float z1, float z2);
    static CSGNode Cone(float r1, float z1, float r2, float z2); 
    static CSGNode Hyperboloid(float r0, float zf, float z1, float z2);
    static CSGNode Box3(float fx, float fy, float fz ); 
    static CSGNode Box3(float fullside); 
    static CSGNode Plane(float nx, float ny, float nz, float d);
    static CSGNode Slab(float nx, float ny, float nz, float d1, float d2 ) ;
    static CSGNode Cylinder(float px, float py, float radius, float z1, float z2) ;
    static CSGNode Disc(float px, float py, float ir, float r, float z1, float z2);

    static CSGNode MakeDemo(const char* name); 
    static CSGNode Make(unsigned typecode, const float* param6, const float* aabb); 

#endif

};


