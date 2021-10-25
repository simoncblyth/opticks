#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "plog/Severity.h"

struct NP ; 

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "saabb.h"
#include "stran.h"


struct CSGName ; 
struct CSGTarget ; 
struct CSGMaker ; 

#include "CSGEnum.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"


#include "CSG_API_EXPORT.hh"

/**
CSGFoundry
============

* CSGSolids contain one or more CSGPrim  (CSGPrim would correspond to Geant4 G4VSolid)
* CSGPrim contain one or more CSGNode    (CSGNode are CSG constituent nodes) 

**/


struct CSG_API CSGFoundry
{
    static const plog::Severity LEVEL ; 
    static const unsigned IMAX ; 
    static CSGFoundry* Load(const char* base, const char* rel);
    static CSGFoundry* Load(const char* dir );
    static int Compare(const CSGFoundry* a , const CSGFoundry* b ); 

    template<typename T>
    static int CompareVec( const char* name, const std::vector<T>& a, const std::vector<T>& b );

    static int CompareBytes(const void* a, const void* b, unsigned num_bytes);


    CSGFoundry();
    void init(); 


    std::string desc() const ;
    std::string descGAS() const ;

    void summary(const char* msg="CSGFoundry::summary") const ;
    std::string descSolids() const ;
    std::string descInst(unsigned ias_idx_, unsigned long long emm=~0ull ) const ;

    void dump() const ;
    void dumpSolid(unsigned solidIdx ) const ;
    int findSolidIdx(const char* label) const  ; // -1 if not found
    void findSolidIdx(std::vector<unsigned>& solid_idx, const char* label) const ; 
    std::string descSolidIdx( const std::vector<unsigned>& solid_selection ) ; 

    void dumpPrim() const ;
    void dumpPrim(unsigned solidIdx ) const ;
    std::string descPrim() const ;
    std::string descPrim(unsigned solidIdx) const  ;

    void dumpNode() const ;
    void dumpNode(unsigned solidIdx ) const ;
    std::string descNode() const ;
    std::string descNode(unsigned solidIdx) const  ;
    std::string descTran(unsigned solidIdx) const  ; 


    AABB iasBB(unsigned ias_idx_, unsigned long long emm=0ull ) const ;
    float4 iasCE(unsigned ias_idx_, unsigned long long emm=0ull ) const;
    void   iasCE(float4& ce, unsigned ias_idx_, unsigned long long emm=0ull ) const;
    void   gasCE(float4& ce, unsigned gas_idx ) const ;
    void   gasCE(float4& ce, const std::vector<unsigned>& gas_idxs ) const ; 

    float  getMaxExtent(const std::vector<unsigned>& solid_selection) const ;
    std::string descSolids(const std::vector<unsigned>& solid_selection) const ;


    CSGPrimSpec getPrimSpec(       unsigned solidIdx) const ;
    CSGPrimSpec getPrimSpecHost(   unsigned solidIdx) const ;
    CSGPrimSpec getPrimSpecDevice( unsigned solidIdx) const ;
    void        checkPrimSpec(     unsigned solidIdx) const ;
    void        checkPrimSpec() const ;


    const CSGSolid*   getSolidByName(const char* name) const ;
    const CSGSolid*   getSolid_(int solidIdx) const ;   // -ve counts from back 
    unsigned          getSolidIdx(const CSGSolid* so) const ; 


    unsigned getNumSolid(int type_) const ;
    unsigned getNumSolid() const;        // STANDARD_SOLID count 
    unsigned getNumSolidTotal() const;   // all solid count 

    unsigned getNumPrim() const;   
    unsigned getNumNode() const;
    unsigned getNumPlan() const;
    unsigned getNumTran() const;
    unsigned getNumItra() const;
    unsigned getNumInst() const;

    const CSGSolid*   getSolid(unsigned solidIdx) const ;  
    const CSGPrim*    getPrim(unsigned primIdx) const ;    
    const CSGNode*    getNode(unsigned nodeIdx) const ;
    const float4*     getPlan(unsigned planIdx) const ;
    const qat4*       getTran(unsigned tranIdx) const ;
    const qat4*       getItra(unsigned itraIdx) const ;
    const qat4*       getInst(unsigned instIdx) const ;

    const CSGPrim*    getSolidPrim(unsigned solidIdx, unsigned primIdxRel) const ;
    const CSGNode*    getSolidPrimNode(unsigned solidIdx, unsigned primIdxRel, unsigned nodeIdxRel) const ;

    void getMeshPrim(std::vector<CSGPrim>& select_prim, unsigned mesh_idx ) const ;
    unsigned getNumMeshPrim(unsigned mesh_idx ) const ;
    std::string descMeshPrim() const ;  



    CSGSolid* addSolid(unsigned num_prim, const char* label, int primOffset_ = -1 );
    CSGPrim*  addPrim(int num_node, int nodeOffset_ ) ;   // former defaults meshIdx:-1, nodeOffset_:-1
    CSGNode*  addNode(CSGNode nd, const std::vector<float4>* pl=nullptr );
    CSGNode*  addNodes(const std::vector<CSGNode>& nds );
    float4*   addPlan(const float4& pl );


    CSGSolid* addDeepCopySolid(unsigned solidIdx, const char* label=nullptr );


    template<typename T> unsigned addTran( const Tran<T>& tr  );
    unsigned addTran( const qat4* tr, const qat4* it ) ;


    //void makeDemoSolids();   // via maker
    CSGSolid* make(const char* name); 


    static void DumpAABB(const char* msg, const float* aabb); 


    void write(const char* dir) const ;
    void write(const char* base, const char* rel) const ;
    void load( const char* base, const char* rel ) ; 
    void load( const char* dir ) ; 

    template<typename T> void loadArray( std::vector<T>& vec, const char* dir, const char* name, bool optional=false ); 

    void upload();
    void inst_find_unique(); 

    unsigned getNumUniqueIAS() const ;
    unsigned getNumUniqueGAS() const ;
    unsigned getNumUniqueINS() const ;

    unsigned getNumInstancesIAS(unsigned ias_idx, unsigned long long emm) const ;
    void     getInstanceTransformsIAS(std::vector<qat4>& select_inst, unsigned ias_idx, unsigned long long emm ) const ;

    unsigned getNumInstancesGAS(unsigned gas_idx) const ;
    void     getInstanceTransformsGAS(std::vector<qat4>& select_inst, unsigned gas_idx ) const ;

    const qat4* getInstanceGAS(unsigned gas_idx_ , unsigned ordinal=0) ;


    // target  
    int getCenterExtent(float4& ce, int midx, int mord, int iidx=-1, qat4* qptr=nullptr ) const ;
    int getTransform(   qat4& q   , int midx, int mord, int iidx=-1) const ;

    // id 
    void parseMOI(int& midx, int& mord, int& iidx, const char* moi) const ; 
    const char* getName(unsigned midx) const ;  


    void kludgeScalePrimBBox( const char* label, float dscale );
    void kludgeScalePrimBBox( unsigned solidIdx, float dscale );

    std::vector<std::string> meshname ;  // meshNames from GGeo/GMeshLib (G4VSolid names from Geant4) should be primName in CF model ?

    std::vector<CSGSolid>  solid ;   
    std::vector<CSGPrim>   prim ; 
    std::vector<CSGNode>   node ; 
    std::vector<float4>    plan ; 
    std::vector<qat4>      tran ;  
    std::vector<qat4>      itra ;  
    std::vector<qat4>      inst ;  

    CSGPrim*    d_prim ; 
    CSGNode*    d_node ; 
    float4*     d_plan ; 
    qat4*       d_itra ; 

    std::vector<unsigned>  ins ; 
    std::vector<unsigned>  gas ; 
    std::vector<unsigned>  ias ; 

    CSGName*    id ; 
    CSGTarget*  target ; 
    CSGMaker*   maker ; 
    bool        deepcopy_everynode_transform ; 

    CSGSolid*   last_added_solid ; 
    CSGPrim*    last_added_prim ; 
    CSGNode*    last_added_node ; 


    NP* bnd ; 
    NP* icdf ; 


};


