#pragma once

#include <optix.h>
#include <vector>
#include "Binding.h"
#include "GAS.h"
#include "IAS.h"

/**
SBT : RG,MS,HG program data preparation 
===========================================

**/
struct PIP ; 
struct Geo ; 

struct SBT 
{
    const PIP*    pip ; 
    Raygen*       raygen ;
    Miss*         miss ;
    HitGroup*     hitgroup ;
    HitGroup*     check ;
 
    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    OptixShaderBindingTable sbt = {};

    std::vector<unsigned> nbis ; 
    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 
    AS*              top ; 


    SBT( const PIP* pip_ ); 
    void setGeo(const Geo* geo); 

    AS* getAS(const char* spec) const ;
    void setTop(const char* spec) ;
    void setTop(AS* top_) ;
    AS* getTop() const ;


    void init();  
    void createRaygen();  
    void updateRaygen();  

    void createMiss();  
    void updateMiss();  

    void createGAS(const Geo* geo);
    void createIAS(const Geo* geo);

    const GAS& getGAS(unsigned gas_idx) const ;
    const IAS& getIAS(unsigned ias_idx) const ;


    void createHitgroup(const Geo* geo);
    void checkHitgroup(); 

    unsigned getNumBI() const ;
    unsigned getNumBI(unsigned gas_idx) const ;
    unsigned getOffsetBI(unsigned gas_idx) const ;
    void     dumpOffsetBI() const ;

    template <typename T>
    static T* UploadArray(const T* array, unsigned num_items ) ; 

    template <typename T>
    static T* DownloadArray(const T* array, unsigned num_items ) ; 
};

