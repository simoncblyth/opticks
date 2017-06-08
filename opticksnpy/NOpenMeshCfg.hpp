#pragma once

#include <string>
#include "NPY_API_EXPORT.hh"

typedef enum
{
    CFG_ZERO       , 
    CFG_CONTIGUOUS , 
    CFG_PHASED     ,
    CFG_SPLIT      , 
    CFG_FLIP       ,
    CFG_NUMFLIP    ,
    CFG_MAXFLIP    ,
    CFG_REVERSED   ,
    CFG_SORTCONTIGUOUS
} NOpenMeshCfgType ;

//  .,+10s/\s*\(CFG\w*\).*$/static const char* \1_ ;/g

struct NPY_API NOpenMeshCfg 
{
    static const char* DEFAULT ; 

    static const char* CFG_ZERO_ ;
    static const char* CFG_CONTIGUOUS_ ;
    static const char* CFG_PHASED_ ;
    static const char* CFG_SPLIT_ ;
    static const char* CFG_FLIP_ ;
    static const char* CFG_NUMFLIP_ ;
    static const char* CFG_MAXFLIP_ ;
    static const char* CFG_REVERSED_ ;
    static const char* CFG_SORTCONTIGUOUS_ ;

    NOpenMeshCfg(const char* cfg=NULL);
    void init();
    void parse(const char* cfg);

    NOpenMeshCfgType  parse_key(const char* k) const ;
    int               parse_val(const char* v) const ;
    std::string       desc(const char* msg="NOpenMeshCfg::desc") const ;


    const char* cfg ; 
    
    int contiguous ; 
    int phased ; 
    int flip ; 
    int split ; 
    int numflip ; 
    int maxflip ; 
    int reversed ; 
    int sortcontiguous ; 
 
};


