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
    AS*                top ; 
 
    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    OptixShaderBindingTable sbt = {};

    std::map<unsigned, GAS> vgas ; 
    std::vector<IAS> vias ; 

    SBT(const PIP* pip_ ); 
    ~SBT(); 

    AS* getAS(const char* spec) const ;
    void setTop(const char* spec) ;
    void setTop(AS* top_) ;
    AS* getTop() const ;

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
    bool isStandardIAS() const ; 
    void createIAS_Standard();
    void createIAS(unsigned ias_idx);
    void createIAS_Selection();
    void createSolidSelectionIAS(unsigned ias_idx, const std::vector<unsigned>& solid_selection);
    void createIAS(const std::vector<qat4>& inst );
    std::string descIAS(const std::vector<qat4>& inst ) const ;

    const IAS& getIAS(unsigned ias_idx) const ;
    const NP*  getIAS_Instances(unsigned ias_idx) const; 


    void createGAS();
    bool isStandardGAS() const ; 
    void createGAS_Standard();
    void createGAS_Selection();
    void createGAS(unsigned gas_idx);
    const GAS& getGAS(unsigned gas_idx) const ;
    std::string descGAS() const ; 

    void setPrimData( HitGroupData& data, const CSGPrim* prim);
    void dumpPrimData( const HitGroupData& data ) const ;
    void checkPrimData( HitGroupData& data, const CSGPrim* prim);

    unsigned getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ; 
    unsigned _getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ;
    unsigned getTotalRec() const ;

};

