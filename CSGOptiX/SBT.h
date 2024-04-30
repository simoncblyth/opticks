#pragma once

/**
SBT : OptiX 7 RG,MS,HG program data preparation 
=================================================

Aim to minimize geometry specifics in here ...

**/

#include <map>
#include <string>
#include <vector>
#include <optix.h>
#include "plog/Severity.h"

#include "Binding.h"
#include "GAS.h"
#include "IAS.h"
#include "sqat4.h"

struct PIP ; 
struct CSGFoundry ; 
struct CSGPrim ; 
struct Properties ; 
struct SScene ; 

//#define WITH_SOPTIX_ACCEL 1 

#ifdef WITH_SOPTIX_ACCEL
struct SOPTIX_Accel ; 
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
    std::vector<SOPTIX_Accel*> vias ; 
#else
    std::map<unsigned, GAS> vgas ; 
    std::vector<IAS> vias ; 
#endif

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
    void createHitgroup();
    void destroyHitgroup();
    void checkHitgroup();

    void createIAS();
    void createIAS(unsigned ias_idx);

    NP* serializeInstances() const ; 
    void collectInstances( const std::vector<qat4>& ias_inst ) ;

    void createIAS(const std::vector<qat4>& inst );
    std::string descIAS(const std::vector<qat4>& inst ) const ;
    OptixTraversableHandle getIASHandle(unsigned ias_idx) const ;
    OptixTraversableHandle getTOPHandle() const ; 

    void createGAS();
    void createGAS(unsigned gas_idx);
    OptixTraversableHandle getGASHandle(unsigned gas_idx) const ;

    std::string descGAS() const ; 

    void setPrimData( HitGroupData& data, const CSGPrim* prim);
    void dumpPrimData( const HitGroupData& data ) const ;
    void checkPrimData( HitGroupData& data, const CSGPrim* prim);

    int getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ; 
    int _getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ;
    unsigned getTotalRec() const ;

};

