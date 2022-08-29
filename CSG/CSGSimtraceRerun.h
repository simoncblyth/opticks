#pragma once

#include <array>
#include <vector>
#include <string>

struct SSim ; 
struct CSGFoundry ; 
struct NP ; 
struct quad4 ; 
struct CSGQuery ; 
struct CSGDraw ; 

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGSimtraceRerun
{ 
    SSim* sim ; 
    const CSGFoundry* fd ; 
    const NP* vv ; 

    const char* SELECTION ; 
    std::vector<int>* selection ; 
    bool with_selection ; 

    const char* fold ; 
    const char* path0 ; 
    const char* path1 ;
 
    NP* simtrace0 ; 
    NP* simtrace1 ; 

    const quad4* qq0 ; 
    quad4*  qq1      ; 

    const CSGQuery* q ; 
    CSGDraw* d ; 

    std::array<unsigned, 5> code_count ; 

    CSGSimtraceRerun(); 

    void init(); 
    std::string desc() const ; 
    static std::string Desc(const quad4& isect1, const quad4& isect0); 

    unsigned intersect_again(quad4& isect1, const quad4& isect0 ); 
    void intersect_again(unsigned idx, bool dump); 
    void intersect_again_selection(unsigned i, bool dump); 
    void intersect_again(); 
    void save() const ; 
    void report() const  ; 

};

