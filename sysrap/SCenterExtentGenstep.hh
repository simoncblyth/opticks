#pragma once

struct NP ; 
struct quad6 ; 
struct float4 ; 
template <typename T> struct Tran ;

#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

/**


NB this was demoted from extg4/X4Intersect as its all generally applicable 
and hence belongs at a lower level 

**/

struct SYSRAP_API SCenterExtentGenstep
{
    static const char* BASE ; 
    static const plog::Severity LEVEL ; 
    static void DumpBoundingBox(const float4& ce, const std::vector<int>& cegs, float gridscale ); 

    SCenterExtentGenstep(const float4* ce_=nullptr); 
    void init(); 
    const char* desc() const ; 
    void dumpBoundingBox() const ; 
    void save(const char* dir) const ; 
    void save() const ; 

    void save_vec(const char* dir, const char* name, const std::vector<quad4>& vv ) const ; 

    template<typename T> void set_meta(const char* key, T value ) ; 


    NP*    gs ;         // not const as need to externally set the meta 
    float  gridscale ;   

    quad4* peta ; 
    bool   dump ; 
    float4 ce ;

    std::vector<int> cegs ; 
    int nx ; 
    int ny ; 
    int nz ; 

    std::vector<int> override_ce ;   

    std::vector<quad4> pp ;
    std::vector<quad4> ii ;

};



