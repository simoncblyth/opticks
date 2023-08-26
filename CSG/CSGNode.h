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


* TODO : it would be convenient for debugging for the lvid to be in the node, 
  for example via bitpack with the boundary or typecode 


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
    |    | co:subNum      | co:subOffset   |                | radius()       |                                                 |
    |    | cx:planeIdx    | cx:planeNum    |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    | zs:zdelta_0    | zs:zdelta_1    | boundary       | index          |                                                 |
    |    | sl:a           | sl:b           |  (1,2)         | (absolute)     |  sl:a,b offsets from origin                     |
    | q1 | cy:z1          | cy:z2          |                | (1,3)          |  cy:z2 > z1                                     |
    |    | ds:z1          | ds:z2          |                |                |                                                 |
    |    | z1()           | z2()           |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |                |                |  q2.w was previously typecode                   |
    |    |                |                |                |                |                                                 |
    | q2 |  BBMin_x       |  BBMin_y       |  BBMin_z       |  BBMax_x       |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |  typecode      | gtransformIdx  |  a.node[:,3,3].view(np.int32) & 0x7fffffff      |
    |    |                |                |  (3,2)         | complement     |                                                 |
    | q3 |  BBMax_y       |  BBMax_z       |                | (3,3)          |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+



typecode gtransformIdx complement
-----------------------------------

* moved typecode from q2.w in order to give 6 contiguous slots for aabb

::

    typecode = cf.node.view(np.uint32)[:,3,2]
    complement = ( cf.node.view(np.uint32)[:,3,3] & 0x80000000 ) >> 31 
    gtransformIdx = cf.node.view(np.uint32)[:,3,3] & 0x7fffffff 

* TODO: complement is naturally related with typecode (not with gtransformIdx)
  so should put them together : could simply flip the sign of the typecode for complemented


gtransformIdx
    integer "pointer" into the tran and itra arrays : containing final "flat" transforms
  
    * final transforms are obtained from the product of structural and CSG transforms 
    * CSGNode within instances : transform product starts inside the outer structural transform. 
    * CSGNode within global remainder : transform product is within root (root always has identity anyhow)
   
    Only leaf nodes use these transforms in csg intersection, intersection onto boolean 
    operator nodes are obtained by choosing between intersects onto leaf nodes : so any transform
    associated with an operator node is ignored  


**Operator nodes pick between intersect distances from their leaf nodes, they never use their own gtransforms.**



Interpret (3,3) zero to mean no transform, so much subtract 1 prior to tran lookup
-----------------------------------------------------------------------------------

::

    In [23]: trIdx = a.node.view(np.int32)[:,3,3] & 0x7fffffff

    In [28]: np.c_[np.unique(trIdx, return_counts=True)]
    Out[28]:
    array([[   0, 8577],
           [   1,    1],
           [   2,    1],
           [   3,    1],
           ...,
           [7389,    1],
           [7390,    1],
           [7391,    1]])

    In [29]: a.tran.shape
    Out[29]: (7391, 4, 4)

    In [31]: a.tran[7391-1]                                                                                                         
    Out[31]: 
    array([[  1. ,   0. ,   0. ,   0. ],
           [  0. ,   1. ,   0. ,   0. ],
           [  0. ,   0. ,   1. ,   0. ],
           [  0. , 831.6,   0. ,   1. ]], dtype=float32)




tree transforms vs final "flat" transforms
--------------------------------------------

Must not conflate these two sets of transforms, as
their nature and considerations for them are very different.

The tree transforms, both structural and CSG, are local to the their nodes 
and are obtained from source geometry. They are held by stree.h/snode.h/snd.hh/sxf.h
The final flat transforms are obtained from the tree transforms via matrix products
(see stree::get_combined_transform). 

The final transforms should not be regarded as modifiable.
Any editing such as for coincidence avoidance nudging needs 
to be done on the tree transforms level prior to flattening, 
eg with CSGImport::importTree. 
This means should NOT be tempted to associate transforms
with every CSGNode in order to allow subsequent modification. 

As the flat transforms are what is on GPU it is beneficial 
to keep the tran/itra arrays as small as possible, eg via unique-ing.
Conversely the tree transforms do not have size concerns. 


subNum subOffset
------------------

Used by compound node types such as CSG_CONTIGUOUS, CSG_DISCONTIGUOUS and the rootnode of boolean trees CSG_UNION/CSG_INTERSECTION/CSG_DIFFERENCE...
Note that because subNum uses q0.u.x and subOffset used q0.u.y this should not (and cannot) be used for leaf nodes. 


**/

struct CSG_API CSGNode
{
    quad q0 ;
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    // only used for CSG_CONVEXPOLYHEDRON and similar prim like CSG_TRAPEZOID which are composed of planes 
    NODE_METHOD unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
    NODE_METHOD unsigned planeNum()      const { return q0.u.y ; } 
    NODE_METHOD void setPlaneIdx(unsigned idx){  q0.u.x = idx ; } 
    NODE_METHOD void setPlaneNum(unsigned num){  q0.u.y = num ; } 

    // used for compound node types such as CSG_CONTIGUOUS, CSG_DISCONTIGUOUS and the rootnode of boolean trees CSG_UNION/CSG_INTERSECTION/CSG_DIFFERENCE...
    NODE_METHOD unsigned subNum()        const { return q0.u.x ; } 
    NODE_METHOD unsigned subOffset()     const { return q0.u.y ; } 

    NODE_METHOD void setSubNum(unsigned num){    q0.u.x = num ; }
    NODE_METHOD void setSubOffset(unsigned num){ q0.u.y = num ; }


    NODE_METHOD void getParam( float& x , float& y , float& z , float& w , float& z1, float& z2 ) const 
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
        if(!p) return ; 
        q0.f.x = *(p+0) ; 
        q0.f.y = *(p+1) ; 
        q0.f.z = *(p+2) ; 
        q0.f.w = *(p+3) ; 
        q1.f.x = *(p+4) ; 
        q1.f.y = *(p+5) ;  
    }

    NODE_METHOD void setParam_Narrow(const double* p)
    { 
        if(!p) return ; 
        q0.f.x = *(p+0) ; 
        q0.f.y = *(p+1) ; 
        q0.f.z = *(p+2) ; 
        q0.f.w = *(p+3) ; 
        q1.f.x = *(p+4) ; 
        q1.f.y = *(p+5) ;  
    }



    NODE_METHOD void getYRange(float& y0, float& y1) const ; 

    NODE_METHOD void setAABBLocal();
    NODE_METHOD void setAABB(  float x0, float y0, float z0, float x1, float y1, float z1){  q2.f.x = x0 ; q2.f.y = y0 ; q2.f.z = z0 ; q2.f.w = x1 ; q3.f.x = y1 ; q3.f.y = z1 ; }  
    NODE_METHOD void setAABB(  float e ){                                                    q2.f.x = -e ; q2.f.y = -e ; q2.f.z = -e ; q2.f.w =  e ; q3.f.x =  e ; q3.f.y =  e ; }  
    NODE_METHOD void setAABB(const float* p)
    { 
        if(!p) return ; 
        q2.f.x = *(p+0) ; 
        q2.f.y = *(p+1) ; 
        q2.f.z = *(p+2) ; 
        q2.f.w = *(p+3) ; 
        q3.f.x = *(p+4) ; 
        q3.f.y = *(p+5) ;  
    }

    NODE_METHOD void setAABB_Narrow(const double* p)
    { 
        if(!p) return ; 
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
    NODE_METHOD void setTransform(  unsigned idx ){     setTransformComplement(idx,  is_complement() )   ; }
    NODE_METHOD void setComplement( bool complement ){  setTransformComplement( gtransformIdx(), complement) ; }
    NODE_METHOD void setTransformComplement( unsigned idx, bool complement ){ q3.u.w = ( idx & 0x7fffffff ) | ( (int(complement) << 31) & 0x80000000) ; }    

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
    std::string tag() const ; 


    std::string brief() const ; 
    static void Dump(const CSGNode* n, unsigned ni, const char* label);  

    bool is_compound() const ; 
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
    static CSGNode BooleanOperator(unsigned op, int num_sub); 

    static CSGNode Overlap(      int num_sub, int sub_offset); 
    static CSGNode Contiguous(   int num_sub, int sub_offset); 
    static CSGNode Discontiguous(int num_sub, int sub_offset); 
    static CSGNode ListHeader(unsigned type, int num_sub, int sub_offset); 


    static CSGNode Zero();
    static CSGNode Sphere(float radius);
    static CSGNode ZSphere(float radius, float z1, float z2);
    static CSGNode Cone(float r1, float z1, float r2, float z2); 
    static CSGNode OldCone(float r1, float z1, float r2, float z2); 
    static CSGNode Hyperboloid(float r0, float zf, float z1, float z2);
    static CSGNode Box3(float fx, float fy, float fz ); 
    static CSGNode Box3(float fullside); 
    static CSGNode Plane(float nx, float ny, float nz, float d);
    static CSGNode Slab(float nx, float ny, float nz, float d1, float d2 ) ;
    static CSGNode Cylinder(   float radius, float z1, float z2) ;
    static CSGNode OldCylinder(float radius, float z1, float z2); 

    static CSGNode InfCylinder(float radius, float hz ) ;
    static CSGNode InfPhiCut(  float startPhi_pi, float deltaPhi_pi ) ;
    static CSGNode InfThetaCut(float startTheta_pi, float deltaTheta_pi ) ; 
    static CSGNode Disc(float px, float py, float ir, float r, float z1, float z2);

    static CSGNode MakeDemo(const char* name); 
    static CSGNode Make(       unsigned typecode ); 
    static CSGNode Make(       unsigned typecode, const float*  param6, const float*  aabb); 
    static CSGNode MakeNarrow( unsigned typecode, const double* param6, const double* aabb); 

#endif

};



#if defined(__CUDACC__) || defined(__CUDABE__)
#else


inline std::ostream& operator<<(std::ostream& os, const CSGNode& n)  
{
    os  
       << std::endl 
       << "q0 " << n.q0 << std::endl
       << "q1 " << n.q1 << std::endl  
       << "q2 " << n.q2 << std::endl
       << "q3 " << n.q3 << std::endl  
       ;   
    return os; 
}

#endif




