#pragma once

#include "squad.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define PRIM_METHOD __device__
#else
   #define PRIM_METHOD 
#endif 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "CSGPrimSpec.h"
#endif

#include "CSG_API_EXPORT.hh"


/**
CSGPrim : contiguous sequence of *numNode* CSGNode starting from *nodeOffset* : complete binary tree of 1,3,7,15,... CSGNode
===============================================================================================================================

* although CSGPrim is uploaded to GPU by CSGFoundry::upload, instances of CSGPrim at first glance 
  appear not to be needed GPU side because the Binding.h HitGroupData carries the same information.  

* But that is disceptive as the uploaded CSGPrim AABB are essential for GAS construction 

* vim replace : shift-R


    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    | q  |      x         |      y         |     z          |      w         |  notes                                          |
    +====+================+================+================+================+=================================================+
    |    |  numNode       |  nodeOffset    | tranOffset     | planOffset     |                                                 |
    | q0 |                |                | TODO:remove    | TODO: remove   |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    | sbtIndexOffset |  meshIdx       | repeatIdx      | primIdx        |                                                 |
    |    |                |                |                |                |                                                 |
    | q1 |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    | q2 |  BBMin_x       |  BBMin_y       |  BBMin_z       |  BBMax_x       |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    | q3 |  BBMax_y       |  BBMax_z       |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+

**/



struct CSG_API CSGPrim  
{
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    // ----  numNode and nodeOffset are fundamental to the meaning of CSGPrim

    PRIM_METHOD int  numNode() const    { return q0.i.x ; } 
    PRIM_METHOD int  nodeOffset() const { return q0.i.y ; } 
    PRIM_METHOD void setNumNode(   int numNode){    q0.i.x = numNode ; }
    PRIM_METHOD void setNodeOffset(int nodeOffset){ q0.i.y = nodeOffset ; }

    // --------- sbtIndex offset is essential for OptiX 7 SBT PrimSpec machinery, but otherwise not relevant to the geometrical meaning  

    PRIM_METHOD unsigned  sbtIndexOffset()    const { return  q1.u.x ; }
    PRIM_METHOD void   setSbtIndexOffset(unsigned sbtIndexOffset){  q1.u.x = sbtIndexOffset ; }
    PRIM_METHOD const unsigned* sbtIndexOffsetPtr() const { return &q1.u.x ; }

    // ---------- ARE tran/plan-offset now just metadata, due to absolute referencing ? 

    PRIM_METHOD int  tranOffset() const { return q0.i.z ; } 
    PRIM_METHOD int  planOffset() const { return q0.i.w ; }

    PRIM_METHOD void setTranOffset(int tranOffset){ q0.i.z = tranOffset ; }
    PRIM_METHOD void setPlanOffset(int planOffset){ q0.i.w = planOffset ; }

    // -------- mesh/repeat/primIdx are metadata for debugging convenience 

    PRIM_METHOD unsigned  meshIdx() const {           return q1.u.y ; }  // aka lvIdx
    PRIM_METHOD void   setMeshIdx(unsigned midx){     q1.u.y = midx ; }

    PRIM_METHOD unsigned repeatIdx() const {          return q1.u.z ; }  // aka solidIdx/GASIdx
    PRIM_METHOD void   setRepeatIdx(unsigned ridx){   q1.u.z = ridx ; }

    PRIM_METHOD unsigned primIdx() const {            return q1.u.w ; }  
    PRIM_METHOD void   setPrimIdx(unsigned pidx){     q1.u.w = pidx ; }

    // ----------  AABB and ce needs changing when transform are applied to the nodes

    PRIM_METHOD void setAABB(  float e  ){                                                   q2.f.x = -e     ; q2.f.y = -e   ; q2.f.z = -e   ; q2.f.w =  e   ; q3.f.x =  e   ; q3.f.y =  e ; }  
    PRIM_METHOD void setAABB(  float x0, float  y0, float z0, float x1, float y1, float z1){      q2.f.x = x0    ; q2.f.y = y0   ; q2.f.z = z0   ; q2.f.w = x1   ; q3.f.x = y1   ; q3.f.y = z1 ; }  
    PRIM_METHOD void getAABB( float& x0, float& y0, float& z0, float& x1, float& y1, float& z1) const { x0 = q2.f.x ; y0 = q2.f.y   ; z0 = q2.f.z   ; x1 = q2.f.w   ; y1 = q3.f.x   ; z1 = q3.f.y ; }
    PRIM_METHOD void setAABB( const float* a){                                               q2.f.x = a[0] ; q2.f.y = a[1] ; q2.f.z = a[2] ; q2.f.w = a[3] ; q3.f.x = a[4] ; q3.f.y = a[5] ; } 
    PRIM_METHOD const float* AABB() const {  return &q2.f.x ; }
    PRIM_METHOD       float* AABB_()      {  return &q2.f.x ; }
    PRIM_METHOD const float3 mn() const {    return make_float3(q2.f.x, q2.f.y, q2.f.z) ; }
    PRIM_METHOD const float3 mx() const {    return make_float3(q2.f.w, q3.f.x, q3.f.y) ; }

    PRIM_METHOD const float4 ce() const 
    {    
        float x0, y0, z0, x1, y1, z1 ; 
        getAABB(x0, y0, z0, x1, y1, z1);  
        return make_float4( (x0 + x1)/2.f,  (y0 + y1)/2.f,  (z0 + z1)/2.f,  extent() );  
    }
    PRIM_METHOD float extent() const 
    {
        float3 d = make_float3( q2.f.w - q2.f.x, q3.f.x - q2.f.y, q3.f.y - q2.f.z ); 
        return fmaxf(fmaxf(d.x, d.y), d.z) /2.f ; 
    }


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void scaleAABB_( float scale )
    {
        float* aabb = AABB_(); 
        for(int i=0 ; i < 6 ; i++ ) *(aabb+i) = *(aabb+i) * scale ; 
    }

    static void Copy(CSGPrim& b, const CSGPrim& a)
    {
        b.q0.f.x = a.q0.f.x ; b.q0.f.y = a.q0.f.y ; b.q0.f.z = a.q0.f.z ; b.q0.f.w = a.q0.f.w ; 
        b.q1.f.x = a.q1.f.x ; b.q1.f.y = a.q1.f.y ; b.q1.f.z = a.q1.f.z ; b.q1.f.w = a.q1.f.w ; 
        b.q2.f.x = a.q2.f.x ; b.q2.f.y = a.q2.f.y ; b.q2.f.z = a.q2.f.z ; b.q2.f.w = a.q2.f.w ; 
        b.q3.f.x = a.q3.f.x ; b.q3.f.y = a.q3.f.y ; b.q3.f.z = a.q3.f.z ; b.q3.f.w = a.q3.f.w ; 
    }

    static int value_offsetof_sbtIndexOffset(){ return offsetof(CSGPrim, q1.u.x)/4 ; }
    static int value_offsetof_AABB(){           return offsetof(CSGPrim, q2.f.x)/4 ; }

    static PRIM_METHOD void select_prim_mesh(const std::vector<CSGPrim>& prims, std::vector<CSGPrim>& select_prims, unsigned mesh_idx_ )
    {
        for(unsigned i=0 ; i < prims.size() ; i++)
        {
            const CSGPrim& pr = prims[i] ; 
            unsigned mesh_idx = pr.meshIdx();  
            if( mesh_idx_ == mesh_idx ) select_prims.push_back(pr) ;
        }
    }
   
    /**
    CSGPrim::select_prim_pointers_mesh
    -----------------------------------

    From the *prims* vector reference find prim with mesh_idx and collect the CSGPrim pointers into *select_prims*   
    **/ 
    static PRIM_METHOD void select_prim_pointers_mesh(const std::vector<CSGPrim>& prims, std::vector<const CSGPrim*>& select_prims, unsigned mesh_idx_ )
    {
        for(unsigned i=0 ; i < prims.size() ; i++)
        {
            const CSGPrim* pr = prims.data() + i ; 
            unsigned mesh_idx = pr->meshIdx();  
            if( mesh_idx_ == mesh_idx ) select_prims.push_back(pr) ;
        }
    }


    static PRIM_METHOD unsigned count_prim_mesh(const std::vector<CSGPrim>& prims, unsigned mesh_idx_ )
    {
        unsigned count = 0u ; 
        for(unsigned i=0 ; i < prims.size() ; i++)
        {
            const CSGPrim& pr = prims[i] ; 
            unsigned mesh_idx = pr.meshIdx();  
            if( mesh_idx_ == mesh_idx ) count += 1 ;
        }
        return count ; 
    }


    std::string desc() const ; 
    static CSGPrimSpec MakeSpec( const CSGPrim* prim0, unsigned primIdx, unsigned numPrim ) ; 
#endif

};


