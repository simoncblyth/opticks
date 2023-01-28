#pragma once
/**
CSGSimtraceSample.h
====================


**/

#include <array>
#include <vector>
#include <string>
#include "plog/Severity.h"

struct SSim ; 
struct CSGFoundry ; 
struct NP ; 
struct quad4 ; 
struct CSGQuery ; 
struct CSGDraw ; 

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGSimtraceSample
{ 
    static const plog::Severity LEVEL ; 

    SSim* sim ; 
    const CSGFoundry* fd ; 
    const NP* vv ; 

    const char* path ; 
    NP* simtrace ; 
    quad4*  qq      ; 

    const CSGQuery* q ; 
    CSGDraw* d ; 

    CSGSimtraceSample(); 
    void init(); 
    std::string desc() const ; 
    int intersect(); 

};

