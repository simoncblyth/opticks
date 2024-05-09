#pragma once
/**
SBT : OptiX 7 RG,MS,HG program data preparation 
=================================================

Aim to minimize geometry specifics in here ...


NOT:WITH_SOPTIX_ACCEL
    ana geometry only using GAS.h IAS.h struct 

WITH_SOPTIX_ACCEL
    ana+tri geometry capability both using SOPTIX_Accel 
    WIP: needs GPU side ana/tri branch and supporting SBT entries

    **FOR NOW : DO NOT COMMIT WITH THIS ENABLED**

**/

#include <map>
#include <string>
#include <vector>
#include <optix.h>
#include "plog/Severity.h"

#include "Binding.h"
#include "sqat4.h"

struct PIP ; 
struct CSGFoundry ; 
struct CSGPrim ; 
struct Properties ; 
struct SScene ; 

#define WITH_SOPTIX_ACCEL 1 

#ifdef WITH_SOPTIX_ACCEL
struct SOPTIX_Accel ; 
struct SOPTIX_MeshGroup ; 
struct SCUDA_MeshGroup ; 
#else
#include "GAS.h"
#include "IAS.h"
#endif


struct SBT 
{
    static const plog::Severity LEVEL ; 

    static bool ValidSpec(const char* spec); 
    std::vector<unsigned>  solid_selection ; 
    unsigned long long emm ; 
    const PIP*      pip ; 
    const Properties* properties ; 
    Raygen*       raygen ;
    Miss*         miss ;
    HitGroup*     hitgroup ;
    HitGroup*     check ;

    const CSGFoundry*  foundry ; 
    const SScene*      scene ; 
 
    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    std::vector<OptixInstance> instances ; 

    OptixShaderBindingTable sbt = {};


#ifdef WITH_SOPTIX_ACCEL
    std::map<unsigned, SOPTIX_Accel*> vgas ; 
    std::map<unsigned, const SOPTIX_MeshGroup*> xgas ; 
    typedef std::map<unsigned, SOPTIX_Accel*>::const_iterator IT ; 
    std::vector<SOPTIX_Accel*> vias ; 
#else
    std::map<unsigned, GAS> vgas ; 
    typedef std::map<unsigned, GAS>::const_iterator IT ; 
    std::vector<IAS> vias ; 
#endif

    static std::string Desc();
    SBT(const PIP* pip_ ); 
    ~SBT(); 


    void init();  
    void destroy(); 

    void createRaygen();  
    void destroyRaygen();  
    void updateRaygen();  

    void createMiss();  
    void destroyMiss();  
    void updateMiss();  

    void setFoundry(const CSGFoundry* foundry); 
    void createGeom();  

    void createGAS();
    void createGAS(unsigned gas_idx);                             // dep. WITH_SOPTIX_ACCEL : create GAS of single CSGSolid and adds to vgas map 
    OptixTraversableHandle getGASHandle(unsigned gas_idx) const ; // dep. WITH_SOPTIX_ACCEL : gets handle from vgas map

    void createIAS();
    void createIAS(unsigned ias_idx);                             // dep. WITH_SOPTIX_ACCEL
    void collectInstances( const std::vector<qat4>& ias_inst ) ;
    NP* serializeInstances() const ; 
    std::string descIAS(const std::vector<qat4>& inst ) const ;
    OptixTraversableHandle getIASHandle(unsigned ias_idx) const ; // dep. WITH_SOPTIX_ACCEL 
    OptixTraversableHandle getTOPHandle() const ; 



    int getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ; 
    int _getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ;  // dep. WITH_SOPTIX_ACCEL : gas/bi/sbt loop with early exit 
    unsigned getTotalRec() const ;                                      // dep. WITH_SOPTIX_ACCEL : gas/bi loop
    std::string descGAS() const ;                                       // dep. WITH_SOPTIX_ACCEL : gas/bi loop  
    void createHitgroup();                                              // dep. WITH_SOPTIX_ACCEL : gas/bi/sbt loop 

    static void UploadHitGroup(OptixShaderBindingTable& sbt, CUdeviceptr& d_hitgroup, HitGroup* hitgroup, size_t tot_rec );

    void destroyHitgroup();
    void checkHitgroup();

    void setPrimData( CustomPrim& cp, const CSGPrim* prim);
    void checkPrimData( CustomPrim& cp, const CSGPrim* prim);
    void dumpPrimData( const CustomPrim& cp ) const ;

#ifdef WITH_SOPTIX_ACCEL
    void setMeshData( TriMesh& tm, const SCUDA_MeshGroup* cmg, int j, int boundary );
#endif

};

