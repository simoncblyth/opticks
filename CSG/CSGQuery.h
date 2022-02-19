#pragma once

#include <string>

struct CSGFoundry ; 
struct CSGPrim ; 
struct CSGNode ; 
struct float4 ; 
struct float3 ; 
struct quad4 ; 
struct qat4 ; 
struct CSGGrid ; 
struct SCanvas ; 

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGQuery 
{
    static const float SD_CUT ; 
    static const int VERBOSE ; 
    static std::string Desc( const quad4& isect, const char* label, bool* valid_intersect=nullptr  ); 

    CSGQuery(const CSGFoundry* fd); 

    void     init(); 
    void     selectPrim(unsigned solidIdx, unsigned primIdxRel );
    void     selectPrim(const CSGPrim* pr );

    int            getSelectedType() const ; 
    int            getSelectedTreeHeight() const ;   // -1: if selected CSGNode is not a tree
    const CSGNode* getSelectedTreeNode( int nodeIdxRel ) const ; 

    const CSGNode* getSelectedNode( int nodeIdxRel ) const ; 

    void     dump(const char* msg) const ;


    void     dumpPrim(const char* msg="CSGQuery::dumpPrim") const ;
    CSGGrid* scanPrim(int resolution) const ;


    float distance(const float3& position ) const ; 
    float operator()(const float3& position) const ;
    void distance( quad4& isect,  const float3& ray_origin ) const ; 

    bool intersect( quad4& isect,  float t_min, const quad4& p ) const ;
    bool intersect( quad4& isect,  float t_min, const float3& ray_origin, const float3& ray_direction, unsigned gsid ) const ;
    bool intersect_again( quad4& isect, const quad4& prev_isect ) const ; 

    static bool IsSpurious( const quad4& isect ); 


    const CSGFoundry* fd ; 
    const CSGPrim* prim0 ; 
    const CSGNode* node0 ; 
    const float4*  plan0 ; 
    const qat4*    itra0 ; 

    const CSGPrim* select_prim ;
    int            select_nodeOffset ; 
    int            select_prim_numNode ;
    const CSGNode* select_root_node ;
    int            select_root_typecode ; 
    int            select_root_subNum ; 
    bool           select_is_tree ; 

 

};


